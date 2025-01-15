from .fedavg import FedAvgEvaluationDevice
from .cocofl import CoCoFLServer, CoCoFLDevice
from sklearn.cluster import KMeans
import logging
import copy
import torch
import numpy as np


class UnitDevice(CoCoFLDevice):
    def __init__(self, device_id):
        super().__init__(device_id)
        self.cluster = None
        self.chunk_index = None
        self.config = None

    def set_config(self, config):
        self.config = config

    #!overrides
    def device_training(self):
        self._check_trainable()
        logging.info(f'[Unit]: Resources: {self.resources}')

        kwargs = copy.deepcopy(self._model_kwargs)
        kwargs.update({'freeze': self.config})

        # reinitialize model and restore state-dict
        state_dict = self._model.state_dict()
        self._model = self._model_class(**kwargs)

        # make sure to use CPU in case quantization is used
        # otherwise push tensors to cuda
        if len(self.config) > 0:
            self._torch_device = 'cpu'
        else:
            for key in state_dict:
                state_dict[key] = state_dict[key].to(self._torch_device)
        print(self._torch_device)
        self._model.load_state_dict(state_dict, strict=False)

        # training
        self._train()
        self._model.to('cpu')



class UnitServer(CoCoFLServer):
    _device_class = UnitDevice
    _device_evaluation_class = FedAvgEvaluationDevice

    def __init__(self, storage_path, n_device_clusters) -> None:
        super().__init__(storage_path)
        self._n_device_clusters = n_device_clusters
        self.configs = []

    def initialize_clusters(self, device_constraints, n_clusters):
        device_constraints_numeric = []

        for resource in device_constraints:
            cur_device_constraints = []
            cur_device_constraints.append(resource.get_time())
            cur_device_constraints.append(resource.get_data())
            cur_device_constraints.append(resource.get_memory())
            device_constraints_numeric.append(cur_device_constraints)

        device_constraints_numeric = np.array(device_constraints_numeric)
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(device_constraints_numeric)

        unit_list = self._model[0].get_units()
        # get list of units from the table
        # run profiling with 3 epochs
        # push
        chunk_indices = []
        for label in np.unique(kmeans.labels_):
            cluster_constraints = device_constraints_numeric[kmeans.labels_ == label]
            min_cluster_resources = np.min(cluster_constraints,axis=0) # has to be minimum at every category
            logging.info(f"Min cluster resources: {min_cluster_resources}")
            for unit in reversed(unit_list):
                if unit == 0:
                    continue
                max_unit_resources = self._model[0].get_max_resources(unit)  # model class is the same for all devices 
                logging.info(f"Max resources (time,data,memory): {max_unit_resources} when training {unit} blocks (units)")
                # if max unit < min cluster for all categories then accept
                if ((max_unit_resources <= min_cluster_resources).all()):
                    cluster_configs = self._model[0].get_freezing_configs_unit(unit)
                    # generate array with relevant configs 
                    self.configs.append(cluster_configs)
                    # generate starting indices list and in initialize assign to each device as chunk index
                    chunk_indices.append(np.resize(np.arange(len(cluster_configs)),len(cluster_constraints)))
                    logging.info(f"Chosen unit for cluster {label}: {unit}")
                    break
        # how to know number of configs per unit? just look at length of config 
        return kmeans.labels_, chunk_indices
    
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

        if self._device_constraints is not None:
            cluster_labels, chunk_indices = self.initialize_clusters(self._device_constraints, self._n_device_clusters)
 
        counter = np.zeros(3,dtype=int)
        for i, device in enumerate(self._devices_list):
            device.set_model(self._model[i], self._model_kwargs[i])
            device.set_train_data(torch.utils.data.Subset(self._train_data.dataset, idxs_list[i]))
            device.lr = self.lr
            device.set_optimizer(self._optimizer, self._optimizer_kwargs)
            device.set_torch_device(self.torch_device)

            if self._device_constraints is not None:
                device.resources = self._device_constraints[i]
                cluster_label = cluster_labels[i]
                device.cluster = cluster_label
                device.chunk_index = chunk_indices[cluster_label][counter[cluster_label]]
                counter[cluster_label] = counter[cluster_label] + 1

        self._devices_list[0].init_model()
        self._global_model = copy.deepcopy(self._devices_list[0]._model.state_dict())
        self._devices_list[0].del_model()

    def shift_chunk_indices(self):
        for device in self._devices_list:
            device.chunk_index = (device.chunk_index + 1) % len(self.configs[device.cluster])
        return
    
    #!overrides
    def init_nn_models(self,idxs):
        for dev_idx in idxs:
            device = self._devices_list[dev_idx]
            device.init_model()
            device.set_model_state_dict(self._global_model)
            device.set_config(self.configs[device.cluster][device.chunk_index]) 
        return
    
    #!overrides
    def post_round(self, round_n, idxs):
        super().post_round(round_n, idxs)
        self.shift_chunk_indices()

