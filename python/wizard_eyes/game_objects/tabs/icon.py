import cv2

from ..game_objects import GameObject


class InterfaceIcon(GameObject):
    """
    Class to represent icons/buttons/items etc. dynamically generated in
    an instance of :class:`TabInterface`.
    """

    PATH_TEMPLATE = '{root}/data/tabs/{name}.npy'

    def __init__(self, name, *args, **kwargs):
        super(InterfaceIcon, self).__init__(*args, **kwargs)
        self.name = name
        self.confidence = None
        self.state = None

    def update(self):
        """
        Run the standard click timer updates, then run template matching to
        determine the current state of the icon. Usually it will have a
        different appearance if activated/clicked.
        """
        super(InterfaceIcon, self).update()

        cur_confidence = -float('inf')
        cur_state = None
        for state, template in self.templates.items():
            mask = self.masks.get(state)
            match = cv2.matchTemplate(
                self.img, template, cv2.TM_CCOEFF_NORMED,
                # TODO: find out why mask of same size causes error
                # mask=mask,
            )
            _, confidence, _, _ = cv2.minMaxLoc(match)

            if confidence > cur_confidence:
                cur_state = state
                cur_confidence = confidence

        self.confidence = cur_confidence
        self.state = cur_state

        # TODO: convert to base class method
        if f'{self.name}_bbox' in self.client.args.show:
            cx1, cy1, _, _ = self.client.get_bbox()
            x1, y1, x2, y2 = self.get_bbox()
            if self.client.is_inside(x1, y1) and self.client.is_inside(x2, y2):
                # convert local to client image
                x1, y1, x2, y2 = self.client.localise(x1, y1, x2, y2)

                # draw a rect around entity on main screen
                cv2.rectangle(
                    self.client.original_img, (x1, y1), (x2, y2),
                    self.colour, 1)
