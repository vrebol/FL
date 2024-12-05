from .fedavg import FedAvgEvaluationDevice
from .cocofl import CoCoFLServer, CoCoFLDevice
from sklearn.cluster import KMeans
import logging
import copy
import torch


class UnitDevice(CoCoFLDevice):
    def __init__(self, device_id):
        super().__init__(self, device_id)
        self.cluster = None
        self.chunk_index = None

    #!overrides
    def device_training(self):
        self._check_trainable()
        logging.info(f'[Unit]: Resources: {self.resources}')

        kwargs = copy.deepcopy(self._model_kwargs)
        kwargs.update({'freeze': self._model_class.get_freezing_config(self.resources)})

        # reinitialize model and restore state-dict
        state_dict = self._model.state_dict()
        self._model = self._model_class(**kwargs)

        # make sure to use CPU in case quantization is used
        # otherwise push tensors to cuda
        if any(self.resources.is_heterogeneous()) == True:
            self._torch_device = 'cpu'
        else:
            for key in state_dict:
                state_dict[key] = state_dict[key].to(self._torch_device)

        self._model.load_state_dict(state_dict, strict=False)

        # training
        self._train()
        self._model.to('cpu')



class UnitServer(CoCoFLServer):
    _device_class = UnitDevice
    _device_evaluation_class = FedAvgEvaluationDevice

    def __init__(self, storage_path, n_device_clusters) -> None:
        super().__init__(self,storage_path)
        self._device_clusters = None
        self._n_device_clusters = n_device_clusters

    #!overrides
    def initialize(self):
        assert not any([self.split_function,
                        self._train_data, self._test_data]) is None, "Uninitialized Values"

        idxs_list = self.split_function(self._train_data, self.n_devices)
        self._evaluation_device = self._device_evaluation_class(0)
        self._evaluation_device.set_model(self._model_evaluation, self._model_evaluation_kwargs)
        self._evaluation_device.init_model()
        self._evaluation_device.set_test_data(self._test_data)
        self._evaluation_device.set_torch_device(self.torch_device)
        self._evaluation_device.is_unbalanced = self.is_unbalanced
        self._evaluation_device.batch_size_test = 512

        self._devices_list = [self._device_class(i) for i in range(self.n_devices)]

        self._devices_list = [self._device_class(i) for i in range(self.n_devices)]

        for i, device in enumerate(self._devices_list):
            device.set_model(self._model[i], self._model_kwargs[i])
            device.set_train_data(torch.utils.data.Subset(self._train_data.dataset, idxs_list[i]))
            device.lr = self.lr
            device.set_optimizer(self._optimizer, self._optimizer_kwargs)
            device.set_torch_device(self.torch_device)

            if self._device_constraints is not None:
                device.resources = self._device_constraints[i]

        self._devices_list[0].init_model()
        self._global_model = copy.deepcopy(self._devices_list[0]._model.state_dict())
        self._devices_list[0].del_model()
