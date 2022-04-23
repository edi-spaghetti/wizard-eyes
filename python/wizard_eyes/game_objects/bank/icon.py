from ...dynamic_menus.icon import AbstractIcon


class BankInterfaceIcon(AbstractIcon):
    """
    Class to represent icons/buttons/items etc. dynamically generated in
    an instance of :class:`BankInterface`.
    """

    PATH_TEMPLATE = '{root}/data/bank/{name}.npy'
