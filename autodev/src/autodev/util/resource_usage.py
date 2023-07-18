import psutil
from threading import Thread
import time
from typing import Dict, Any, Hashable, Optional

import numpy as np
from GPUtil import GPUtil


class ResourceUsageData:
    def __init__(self, base_system_memory_usage=0, gpu_index: Optional[int] = 0):
        self._gpu_index = gpu_index
        self._base_system_memory_usage = base_system_memory_usage
        self._calls = 0
        self._system_memory_usage_values_mb = []
        self._cpu_utilization_values_percent = []
        self._gpu_memory_usage_values_mb = []
        self._gpu_utilization_values_percent = []

    def add_data_point(self):

        cpu_util = psutil.cpu_percent()
        if self._calls > 0:  # first value is 0 and should be ignored
            self._cpu_utilization_values_percent.append(cpu_util)
        mem_used_mb = psutil.virtual_memory().used / 1000000
        self._system_memory_usage_values_mb.append(mem_used_mb)

        if self._gpu_index is not None:
            gpus = GPUtil.getGPUs()
            if self._gpu_index < len(gpus):
                gpu = gpus[self._gpu_index]
                self._gpu_memory_usage_values_mb.append(gpu.memoryUsed)
                self._gpu_utilization_values_percent.append(gpu.load * 100)

        self._calls += 1

    def system_memory_usage_values_mb(self) -> np.ndarray:
        return np.array(self._system_memory_usage_values_mb)

    def cpu_utilization_values_percent(self) -> np.ndarray:
        return np.array(self._cpu_utilization_values_percent)

    def gpu_utilization_values_percent(self) -> np.ndarray:
        return np.array(self._gpu_utilization_values_percent)

    def gpu_memory_usage_values(self) -> np.ndarray:
        return np.array(self._gpu_memory_usage_values_mb)

    def system_memory_usage_mb(self, agg=np.median):
        return agg(self.system_memory_usage_values_mb())

    def cpu_utilization_percent(self, agg=np.median):
        return agg(self._cpu_utilization_values_percent)

    def gpu_memory_usage_mb(self, agg=np.median):
        if len(self._gpu_memory_usage_values_mb) > 0:
            return agg(self.gpu_memory_usage_values())
        return np.nan

    def gpu_utilization_percent(self, agg=np.median):
        if len(self._gpu_utilization_values_percent) > 0:
            return agg(self._gpu_utilization_values_percent)
        return np.nan


class ResourceUsageMonitoringThread(Thread):
    def __init__(self, period_ms=100, use_base_system_memory_usage=True, gpu_index: Optional[int] = 0):
        super().__init__(daemon=True)
        self.period_ms = period_ms
        self._usage_data: Optional[ResourceUsageData] = None
        self.usage_data_dict: Dict[Any, ResourceUsageData] = {}
        self._gpu_index = gpu_index
        if use_base_system_memory_usage:
            self._base_system_memory_usage = psutil.virtual_memory().used
        else:
            self._base_system_memory_usage = 0

    def begin_collection(self, key: Hashable):
        self._usage_data = ResourceUsageData(base_system_memory_usage=self._base_system_memory_usage,
            gpu_index=self._gpu_index)
        self.usage_data_dict[key] = self._usage_data

    def end_collection(self) -> ResourceUsageData:
        result = self._usage_data
        self._usage_data = None
        return result

    def run(self):
        while True:
            if self._usage_data is not None:
                self._usage_data.add_data_point()
            time.sleep(self.period_ms/1000)