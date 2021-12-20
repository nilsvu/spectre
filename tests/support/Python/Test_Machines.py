# Distributed under the MIT License.
# See LICENSE.txt for details.

import unittest
from spectre.support.Machines import (all_machines, this_machine,
                                      UnknownMachineError)


class TestMachines(unittest.TestCase):
    def test_machine_names_and_keys(self):
        for name, machine in all_machines.items():
            self.assertEqual(name, machine.Name)

    def test_this_machine(self):
        with self.assertRaises(UnknownMachineError):
            this_machine("garbage.hostname")
        self.assertEqual(this_machine("login01.cluster").Name, 'Minerva')
        self.assertEqual(this_machine("login02.cluster").Name, 'Minerva')
        self.assertEqual(this_machine("node004.cluster").Name, 'Minerva')


if __name__ == '__main__':
    unittest.main(verbosity=2)
