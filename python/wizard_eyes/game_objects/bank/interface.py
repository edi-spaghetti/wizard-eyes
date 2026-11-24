from typing import Dict, List, Optional

from . import grid
from ...dynamic_menus.interface import AbstractInterface
from ..template import Template
from .icon import BankInterfaceIcon
from .quantity import TEXT_BOX, read_quantity


class BankInterface(AbstractInterface):
    """Bank contents as a grid of slots (see :mod:`.grid`).

    Instead of a hand-painted alpha image, every visible slot is an icon,
    each with its own colour (:func:`grid.slot_colour`). The number of rows
    follows the bank's height: slots stop above the bottom button row. A
    bot says what it expects in which slot with :meth:`assign_slot`; other
    slots are named ``slot<i>`` and only tell ``something`` from
    ``nothing``.
    """

    def __init__(self, client, widget):
        super().__init__(client, widget)
        self.slot_names: Dict[int, str] = {}
        self.slots: Dict[int, BankInterfaceIcon] = {}
        self.rows = 0

    @property
    def icon_class(self):
        return BankInterfaceIcon

    def assign_slot(self, index: int, name: str, templates: List[str]):
        """Expect ``name`` in slot ``index`` (row * 8 + column, from the
        top left of the tag tab), identified by these templates (e.g.
        ``['grimy_irit', 'grimy_irit_placeholder']``). Call before the bank
        is located."""
        group = self.create_template_group(name, grid.slot_colour(index), 1)
        for template in templates:
            self.add_template_to_group(group, Template(template))
        self.slot_names[index] = name
        return group

    def slot(self, index: int) -> Optional[BankInterfaceIcon]:
        return self.slots.get(index)

    def quantity(self, index: int) -> Optional[int]:
        """How many of its item slot ``index`` holds (0 for a placeholder),
        None if there's no readable number (see :mod:`.quantity`)."""
        icon = self.slots.get(index)
        if icon is None:
            return None
        x, y, _, _ = icon.get_bbox()
        x1, y1, x2, y2 = TEXT_BOX
        img = self.client.get_img_at((x + x1, y + y1, x + x2, y + y2),
                                     mode=self.client.BGRA)
        if img is None or not img.size:
            return None
        return read_quantity(img)

    def locate_icons_by_alpha(self):
        """Build the slot icons from the bank's position and the bottom
        button row (called once the interface is located)."""
        bank = self.widget.parent
        if not bank.located or bank.buttons_top is None:
            return False
        bx1, by1, _, _ = bank.get_bbox()
        self.rows = grid.slot_rows(bank.buttons_top)
        self.icons = dict()
        self.slots = dict()
        for i in range(self.rows * grid.COLUMNS):
            x1, y1, x2, y2 = grid.slot_bbox(i)
            colour = grid.slot_colour(i)
            assigned = self.slot_names.get(i)
            name = assigned or f'slot{i}'
            icon = self.icon_class(name, self.client, self,
                                   type_=assigned or 'slot')
            icon.slot = i
            icon.DEFAULT_COLOUR = colour
            icon.set_aoi(bx1 + x1, by1 + y1, bx1 + x2, by1 + y2)
            if assigned:
                group = self.template_group_from_colour(colour)
                templates = [t.name for t in group.templates]
                icon.load_templates(templates)
                icon.load_masks(templates)
                for template in group.templates:
                    if template.alias:
                        icon.load_masks([template.alias])
                        icon.alias_mask(template.alias, template.name)
            icon.DETECT_ANYTHING = True
            self.icons[name] = icon
            self.slots[i] = icon
            setattr(self, name, icon)
        missing = [i for i in self.slot_names if i not in self.slots]
        if missing:
            self.logger.warning(
                f'Assigned slots {missing} are not visible: only '
                f'{self.rows} rows fit in the bank at this window size')
        self.logger.info(f'Bank grid: {self.rows} rows x {grid.COLUMNS} '
                         f'columns, {len(self.slot_names)} assigned')
        return True
