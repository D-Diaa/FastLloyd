"""
MPI-based communication module for distributed computing applications.

This module provides a high-level interface for MPI (Message Passing Interface) communications,
with additional features for simulating network delays and tracking communication statistics.
"""

import sys
import time

from mpi4py import MPI


class Communicator:
    """
    A wrapper class for MPI communications with additional functionality for delays and statistics.

    This class provides methods for point-to-point and collective communications between processes,
    with optional artificial delays to simulate network latency. It also tracks communication
    statistics such as the number of communication rounds and total data size transferred.

    Attributes:
        comm (MPI.Comm): The main MPI communicator (MPI.COMM_WORLD)
        world_size (int): Total number of processes
        rank (int): Rank of the current process
        root (int): Rank of the root process (default: 0)
        delay (float): Artificial delay in seconds for communications
        client_group (MPI.Group): MPI group excluding the root process
        client_comm (MPI.Comm): MPI communicator for client-only communications
        num_comm_rounds (int): Number of communication rounds performed
        comm_size (int): Total size of data communicated in bytes
    """

    def __init__(self, delay=0.05):
        """
        Initialize the Communicator with specified delay.

        Args:
            delay (float, optional): Artificial delay in seconds. Defaults to 0.05.
        """
        self.comm = MPI.COMM_WORLD
        self.world_size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()
        self.root = 0
        self.delay = delay

        # Create a new group excluding the root
        self.client_group = self.comm.Get_group().Excl([self.root])
        # Create a new communicator from the new group
        self.client_comm = self.comm.Create(self.client_group)

        # Initialize communication statistics
        self.num_comm_rounds = 0
        self.comm_size = 0

    def set_delay(self, delay):
        """
        Set the artificial communication delay.

        Args:
            delay (float): New delay value in seconds
        """
        self.delay = delay

    def send(self, obj, dest, group="all"):
        """
        Send an object to a destination process.

        Args:
            obj: Object to send
            dest (int): Rank of the destination process
            group (str, optional): Communication group ("all" or "clients"). Defaults to "all".
        """
        if group == "all":
            self.comm.send(obj, dest=dest)
        elif group == "clients":
            self.client_comm.send(obj, dest=dest)

    def send_delay(self, obj, dest, group="all"):
        """
        Send an object with artificial delay and update statistics.

        Args:
            obj: Object to send
            dest (int): Rank of the destination process
            group (str, optional): Communication group ("all" or "clients"). Defaults to "all".
        """
        time.sleep(self.delay)
        self.num_comm_rounds += 1
        self.comm_size += obj.nbytes
        self.send(obj, dest, group)

    def recv(self, source, group="all"):
        """
        Receive an object from a source process.

        Args:
            source (int): Rank of the source process
            group (str, optional): Communication group ("all" or "clients"). Defaults to "all".

        Returns:
            The received object
        """
        if group == "all":
            return self.comm.recv(source=source)
        elif group == "clients":
            return self.client_comm.recv(source=source)

    def recv_delay(self, source, group="all"):
        """
        Receive an object and update statistics.

        Args:
            source (int): Rank of the source process
            group (str, optional): Communication group ("all" or "clients"). Defaults to "all".

        Returns:
            The received object
        """
        self.num_comm_rounds += 1
        return self.recv(source, group)

    def gather(self, obj, root, group="all"):
        """
        Gather objects from all processes to the root process.

        Args:
            obj: Object to contribute to the gather operation
            root (int): Rank of the root process
            group (str, optional): Communication group ("all" or "clients"). Defaults to "all".

        Returns:
            List of gathered objects on root process, None on other processes
        """
        if group == "all":
            return self.comm.gather(obj, root=root)
        elif group == "clients":
            return self.client_comm.gather(obj, root=root)

    def gather_delay(self, obj, root, group="all"):
        """
        Gather objects with artificial delay and update statistics.

        Args:
            obj: Object to contribute to the gather operation
            root (int): Rank of the root process
            group (str, optional): Communication group ("all" or "clients"). Defaults to "all".

        Returns:
            List of gathered objects on root process, None on other processes
        """
        self.comm_size += obj.nbytes if root != self.rank else 0
        self.num_comm_rounds += 1
        time.sleep(self.delay)
        return self.gather(obj, root, group)

    def allgather(self, obj, group="all"):
        """
        Gather objects from all processes to all processes.

        Args:
            obj: Object to contribute to the allgather operation
            group (str, optional): Communication group ("all" or "clients"). Defaults to "all".

        Returns:
            List of gathered objects from all processes
        """
        if group == "all":
            return self.comm.allgather(obj)
        elif group == "clients":
            return self.client_comm.allgather(obj)

    def allgather_delay(self, obj, group="all"):
        """
        Gather objects to all processes with artificial delay and update statistics.

        Args:
            obj: Object to contribute to the allgather operation
            group (str, optional): Communication group ("all" or "clients"). Defaults to "all".

        Returns:
            List of gathered objects from all processes
        """
        self.comm_size += obj.nbytes * (self.world_size - 1)
        self.num_comm_rounds += 1
        time.sleep(self.delay)
        return self.allgather(obj, group)

    def bcast(self, obj, root, group="all"):
        """
        Broadcast an object from the root process to all processes.

        Args:
            obj: Object to broadcast (only meaningful on root process)
            root (int): Rank of the root process
            group (str, optional): Communication group ("all" or "clients"). Defaults to "all".

        Returns:
            The broadcast object
        """
        if group == "all":
            return self.comm.bcast(obj, root=root)
        elif group == "clients":
            return self.client_comm.bcast(obj, root=root)

    def bcast_delay(self, obj, root, group="all"):
        """
        Broadcast an object with artificial delay and update statistics.

        Args:
            obj: Object to broadcast (only meaningful on root process)
            root (int): Rank of the root process
            group (str, optional): Communication group ("all" or "clients"). Defaults to "all".

        Returns:
            The broadcast object
        """
        self.comm_size += obj.nbytes * (self.world_size - 1) if self.rank == root else 0
        self.num_comm_rounds += 1
        time.sleep(self.delay)
        return self.bcast(obj, root, group)

    def get_comm_stats(self):
        """
        Get current communication statistics.

        Returns:
            dict: Dictionary containing number of communication rounds and total data size
        """
        return {
            'num_comm_rounds': self.num_comm_rounds,
            'comm_size': self.comm_size
        }

    def print_comm_stats(self):
        """
        Print current communication statistics.

        Returns:
            str: Formatted string containing rank and communication statistics
        """
        to_print = f"{self.rank}, {self.get_comm_stats()}"
        print(to_print)
        sys.stdout.flush()
        return to_print

    def reset_comm_stats(self):
        """Reset communication statistics counters."""
        self.num_comm_rounds = 0
        self.comm_size = 0

    def close(self):
        """Clean up MPI resources by freeing the client group and communicator."""
        self.client_group.Free()
        if self.client_comm != MPI.COMM_NULL:
            self.client_comm.Free()


# Global communicator instance
comm = Communicator()


def debug(obj=None):
    """
    Print debug information with process rank.

    Args:
        obj: Object to print (optional)
    """
    print(comm.rank, obj)
    sys.stdout.flush()


def fail_together(fn, error_message):
    """
    Execute a function and ensure all processes fail if any process fails.

    This function executes the provided function and uses allgather to check if any
    process encountered an error. If any process fails, all processes will raise
    the same exception.

    Args:
        fn: Function to execute
        error_message (str): Error message to use if failure occurs

    Returns:
        The result of fn() if successful

    Raises:
        Exception: If any process fails, with the provided error message
    """
    feasible = True
    try:
        x = fn()
    except Exception as e:
        debug(e)
        feasible = False
    # to let all parties now if a client failed
    # Without delay cause this is not part of the protocol
    # In practice, when a party fails, everyone will timeout and ignore
    feasibles = comm.allgather(feasible)
    if not all(feasibles):
        raise Exception(error_message)
    return x