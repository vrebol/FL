from .fedavg import FedAvgDevice, FedAvgEvaluationDevice, FedAvgServer
import logging
import copy
import torch
import numpy as np


class CoCoFLDevice(FedAvgDevice):
    def __init__(self, device_id):
        super().__init__(device_id)
        self._block_selection = []
        self._cluster_selection = []

    #!overrides
    def device_training(self):
        self._check_trainable()
        logging.info(f'[COCOFL]: Resources: {self.resources}')

        freezing_config = self._model_class.get_freezing_config(self.resources)
        kwargs = copy.deepcopy(self._model_kwargs)
        kwargs.update({'freeze': freezing_config})

        # reinitialize model and restore state-dict
        state_dict = self._model.state_dict()
        self._model = self._model_class(**kwargs)

        model_len = self._model.n_freezable_layers()

        # store block selection of current device
        block_selections = np.zeros(model_len,dtype=int) 
        block_selections[freezing_config] = 1
        self._block_selection = 1 - block_selections 

        # store cluster selection of current device
        cluster_selections = np.zeros(model_len,dtype=int) 
        cluster_selections[model_len-len(freezing_config)-1] = 1
        self._cluster_selection = cluster_selections

        # make sure to use CPU in case quantization is used
        # otherwise push tensors to cuda
        if len(freezing_config) > 0:
            self._torch_device = 'cpu'
        else:
            for key in state_dict:
                state_dict[key] = state_dict[key].to(self._torch_device)

        self._model.load_state_dict(state_dict, strict=False)

        # training
        self._train()
        self._model.to('cpu')

    def _train(self, n_epochs=1):
        self._model.to(self._torch_device)
        self._model.train()
        trainloader = torch.utils.data.DataLoader(self._train_data,
                                                  batch_size=self._batch_size_train,
                                                  shuffle=True)

        optimizer = self._optimizer(filter(lambda x: x.requires_grad, self._model.parameters()), lr=self.lr, **self._optimizer_kwargs)

        for _ in range(n_epochs):
            for _, (inputs, labels) in enumerate(trainloader):
                inputs, labels = inputs.to(self._torch_device), labels.to(self._torch_device)

                optimizer.zero_grad()
                output = self._model(inputs)
                loss = torch.nn.functional.cross_entropy(output, labels)

                self.assert_if_nan(loss)
                loss.backward()
                optimizer.step()

    #!overrides
    # only return items in state dict to the server that are trained (changed it's value)
    def get_model_state_dict(self):
        state_dict = copy.deepcopy(super().get_model_state_dict())
        for name, param in self._model.named_parameters():
            if not param.requires_grad:
                state_dict.pop(name)
        return state_dict


class CoCoFLServer(FedAvgServer):
    _device_class = CoCoFLDevice
    _device_evaluation_class = FedAvgEvaluationDevice

    def __init__(self, device_id):
        super().__init__(device_id)
        self._measurements_dict['block_selections'] = []
        self._measurements_dict['cluster_selections'] = []


    #!overrides
    @staticmethod
    def model_averaging(list_of_state_dicts, eval_device_dict=None):
        averaging_exceptions = ['num_batches_tracked']

        if eval_device_dict is not None:
            averaged_dict = copy.deepcopy(eval_device_dict)
        else:
            averaged_dict = copy.deepcopy(list_of_state_dicts[0])

        for key in averaged_dict:
            if all(module_name not in key for module_name in averaging_exceptions):

                list_of_averageable_params = [state_dict[key] for state_dict in list_of_state_dicts if key in state_dict]
                r = len(list_of_averageable_params)/len(list_of_state_dicts)
                if len(list_of_averageable_params) == 0:
                    continue
                else:
                    # special treatment for op_scale and op_scale bw
                    if key.endswith('op_scale') or key.endswith('op_scale_bw'):
                        averaged_dict[key] = torch.atleast_1d(torch.mean(torch.stack(list_of_averageable_params), dim=0))
                    else:
                        # Averaging of trained layers according to Algorithm2 in the paper
                        averaged_dict[key] = r * torch.mean(torch.stack(list_of_averageable_params), dim=0) + (1 - r)*averaged_dict[key]

        averaged_dict = {k: v for k, v in averaged_dict.items() if all(module_name not in k for module_name in averaging_exceptions)}
        return averaged_dict
    
    #!overrides
    def initialize(self):
        super().initialize()
        self._measurements_dict['block_selections_small'] = np.zeros(self._model[0].n_freezable_layers())
        self._measurements_dict['block_selections_medium'] = np.zeros(self._model[0].n_freezable_layers())
        self._measurements_dict['block_selections_large']= np.zeros(self._model[0].n_freezable_layers())
        self._measurements_dict['block_selections']= np.zeros(self._model[0].n_freezable_layers())


        self._measurements_dict['cluster_selections_small'] = np.zeros(self._model[0].n_freezable_layers())
        self._measurements_dict['cluster_selections_medium'] = np.zeros(self._model[0].n_freezable_layers())
        self._measurements_dict['cluster_selections_large'] = np.zeros(self._model[0].n_freezable_layers())
        self._measurements_dict['cluster_selections']= np.zeros(self._model[0].n_freezable_layers())


    #!overrides
    def post_round(self, round_n, idxs):
        super().post_round(round_n, idxs)

        used_devices = [self._devices_list[i] for i in idxs]
        for device in used_devices:
            self._measurements_dict['block_selections'] += device._block_selection
            self._measurements_dict['cluster_selections'] += device._cluster_selection

            if device._size == 0:
                self._measurements_dict['block_selections_small'] += device._block_selection
                self._measurements_dict['cluster_selections_small'] += device._cluster_selection
            elif device._size == 1:
                self._measurements_dict['block_selections_medium'] += device._block_selection
                self._measurements_dict['cluster_selections_medium'] += device._cluster_selection
            elif device._size == 2:
                self._measurements_dict['block_selections_large'] += device._block_selection
                self._measurements_dict['cluster_selections_large'] += device._cluster_selection

            
        self._measurements_dict['block_selections'] = list(self._measurements_dict['block_selections'])
        self._measurements_dict['block_selections_small'] = list(self._measurements_dict['block_selections_small'])
        self._measurements_dict['block_selections_medium'] = list(self._measurements_dict['block_selections_medium'])
        self._measurements_dict['block_selections_large'] = list(self._measurements_dict['block_selections_large'])
        self._measurements_dict['cluster_selections'] = list(self._measurements_dict['cluster_selections'])
        self._measurements_dict['cluster_selections_small'] = list(self._measurements_dict['cluster_selections_small'])
        self._measurements_dict['cluster_selections_medium'] = list(self._measurements_dict['cluster_selections_medium'])
        self._measurements_dict['cluster_selections_large'] = list(self._measurements_dict['cluster_selections_large'])