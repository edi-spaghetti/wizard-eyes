from ...dynamic_menus.icon import AbstractIcon


class ChatInterfaceIcon(AbstractIcon):
    """
    Class to represent icons/buttons/items etc. dynamically generated in
    an instance of :class:`ChatInterface`.
    """

    PATH_TEMPLATE = '{root}/data/chat/{name}.npy'
