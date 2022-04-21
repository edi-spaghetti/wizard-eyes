from ...dynamic_menus.icon import AbstractIcon


class InterfaceIcon(AbstractIcon):
    """
    Class to represent icons/buttons/items etc. dynamically generated in
    an instance of :class:`TabInterface`.
    """

    PATH_TEMPLATE = '{root}/data/tabs/{name}.npy'
