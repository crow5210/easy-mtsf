import logging

import torch
from torch.utils.tensorboard import SummaryWriter


class AvgMeter(object):
    """Average meter.
    """

    def __init__(self):
        self._sum = 0.
        self._count = 0

    def reset(self):
        """Reset counter.
        """

        self._sum = 0.
        self._count = 0

    def update(self, value: float, n: int = 1):
        """Update sum and count.

        Args:
            value (float): value.
            n (int): number.
        """

        self._sum += value * n
        self._count += n

    @property
    def avg(self) -> float:
        """Get average value.

        Returns:
            avg (float)
        """

        return self._sum / self._count if self._count != 0 else 0


class MeterPool:
    """Meter container
    """

    def __init__(self):
        self._pool = {}

    def register(self, name: str, fmt: str = '{:f}', plt: bool = True):
        """Init an average meter and add it to meter pool.

        Args:
            name (str): meter name (must be unique).
            fmt (str): meter output format.
            plt (bool): set ```True``` to plot it in tensorboard
                when calling ```plt_meters```.
        """

        if name in self._pool:
            raise ValueError(f'Meter {name} already existed.')

        self._pool[name] = {
            'meter': AvgMeter(),
            'index': len(self._pool.keys()),
            'format': fmt,
            'plt': plt
        }

    def update(self, name: str, value: float, n: int = 1):
        """Update average meter.

        Args:
            name (str): meter name.
            value (str): value.
            n: (int): num.
        """

        self._pool[name]['meter'].update(value, n)

    def get_avg(self, name: str) -> float:
        """Get average value.

        Args:
            name (str): meter name.

        Returns:
            avg (float)
        """

        return self._pool[name]['meter'].avg

    def print_meters(self, logger: logging.Logger = None):
        """Print the specified type of meters.

        Args:
            logger (logging.Logger): logger
        """

        print_list = []
        for i in range(len(self._pool.keys())):
            for name, value in self._pool.items():
                if value['index'] == i:
                    print_list.append(
                        ('{}: ' + value['format']).format(name, value['meter'].avg)
                    )
        print_str = 'Result : [{}]'.format(', '.join(print_list))
        if logger is None:
            print(print_str)
        else:
            logger.info(print_str)

    def plt_meters(self, step: int, tensorboard_writer: SummaryWriter):
        """Plot the specified type of meters in tensorboard.

        Args:
            step (int): Global step value to record
            tensorboard_writer (SummaryWriter): tensorboard SummaryWriter
        """

        for name, value in self._pool.items():
            if value['plt'] and tensorboard_writer is not None:
                tensorboard_writer.add_scalar(name, value['meter'].avg, global_step=step)
                tensorboard_writer.flush()

    def reset(self):
        """Reset all meters.
        """

        for _, value in self._pool.items():
            value['meter'].reset()