from ...dynamic_menus.interface import AbstractInterface
from .icon import BankInterfaceIcon


class BankInterface(AbstractInterface):

    def __init__(self, client, widget):
        super().__init__(client, widget)

    @property
    def icon_class(self):
        return BankInterfaceIcon
