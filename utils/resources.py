import torch
import numpy as np

class Constant():
    def __init__(self, const_value):
        self.value = const_value
        pass

    def is_heterogeneous(self):
        if self.value != 1.0:
            return True
        else:
            return False

    def __call__(self):
        return self.value

    def __repr__(self):
        return f"c:{self.value}"


class Uniform():
    def __init__(self, low, high):
        self.low = low
        self.high = high
        assert self.high > self.low, "Error in uniform"

    def is_heterogeneous(self):
        return True

    def __call__(self):
        res = float((self.high - self.low)*torch.rand(1) + self.low)
        return res

    def __repr__(self):
        return f"u:{self.low}/{self.high}"


class DeviceResources():
    def __init__(self):
        self._time_function = None
        self._data_function = None
        self._memory_function = None

    def set_all(self, function_time, function_data, function_memory):
        self.set_time_selection_F(function_time)
        self.set_data_selection_F(function_data)
        self.set_memory_selection_F(function_memory)

    def set_time_selection_F(self, function):
        self._time_function = function

    def set_data_selection_F(self, function):
        self._data_function = function

    def set_memory_selection_F(self, function):
        self._memory_function = function

    def get_time(self):
        return self._time_function()

    def get_data(self):
        return self._data_function()

    def get_memory(self):
        return self._memory_function()

    def __repr__(self) -> str:
        out = "[" + "t->" + self._time_function.__str__()
        out += " d->" + self._data_function.__str__()
        out += " m->" + self._memory_function.__str__() + "]"
        return out

    def is_heterogeneous(self):
        res = [self._time_function.is_heterogeneous(),
                self._data_function.is_heterogeneous(),
                self._memory_function.is_heterogeneous()]
        return res

def createDeviceResources(device_constraints, min_resources, distr="Uniform"):
    if not distr in ["Uniform","Normal"]:
        return None
    
    if distr == "Normal":
        resources_a = np.zeros((100,3))
        cnt = 0
        for r in min_resources:
            rng = np.random.default_rng(seed=7)
            loc = r + ((1.2 - r)/2)
            scale = ((1.2 - r)/2)/2
            resources = rng.normal(loc=loc,scale=scale,size=(100))
            resources[resources < r] = r
            resources[::-1].sort()
            resources_a[:,cnt] = resources 
            cnt = cnt + 1
    else:
        resources_a = np.zeros((100,3))
        cnt = 0
        for r in min_resources:
            rng = np.random.default_rng(seed=7)
            resources = rng.uniform(low=r,high=1.2,size=(100))
            resources[::-1].sort()
            resources_a[:,cnt] = resources 
            cnt = cnt + 1
    cnt = 0
    for resource in device_constraints:
        resource.set_all(Constant(resources_a[cnt][0]), Constant(resources_a[cnt][1]), Constant(resources_a[cnt][2]))
        cnt = cnt + 1

    return device_constraints
