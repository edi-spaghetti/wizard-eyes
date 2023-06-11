import argparse

from wizard_eyes.application import Application
from wizard_eyes.constants import REDA
from wizard_eyes.game_objects.template import Template


class TestContainersApp(Application):
    """Example application that tests the dynamic menus are working correctly.
    """

    def create_parser(self) -> argparse.ArgumentParser:
        parser = super().create_parser()

        params = ('name', 'alias', 'threshold', 'equip')
        parser.add_argument('--bank-tab', default='tabINF',
                            help='Name of the bank tab widget to use.')
        parser.add_argument(
            '--template',
            default=[['coins_10000', 'None', 'None', '0']],
            nargs=len(params), metavar=params,
            action='append',
            help='Templates to load into the bank, '
                 'inventory and equipment tabs. Alias, can be "None" or an'
                 'alias name. Threshold can be "None" or a numer. Equip '
                 'should be zero or one.'
        )
        return parser

    def setup(self):
        """Set the menus that we care about to auto locate,
        and create templates for all of them based on command line args."""

        # this is a live demo, always show bboxes and state to avoid confusion
        self.client.args.show.update({'mouse', '*bbox', '*state'})

        inv = self.client.tabs.inventory.interface
        bank = getattr(self.client.bank, self.args.bank_tab).interface
        eq = self.client.tabs.equipment.interface

        getattr(self.client.bank, self.args.bank_tab).auto_locate = True
        self.client.tabs.inventory.auto_locate = True
        self.client.tabs.equipment.auto_locate = True
        self.client.chat.all.auto_locate = True

        # set up bank templates
        for name, alias, threshold, _ in self.args.template:
            group = bank.create_template_group(
                name, REDA, len(self.args.template))

            alias = None if alias == 'None' else alias
            threshold = (Template.threshold
                         if threshold == 'None' else float(threshold))

            template = Template(name, alias=alias, threshold=threshold)
            placeholder = Template(
                f'{name}_placeholder', alias=alias, threshold=threshold)

            bank.add_template_to_group(group, template)
            bank.add_template_to_group(group, placeholder)

        # set up inventory templates
        inventory = [name for name, _, _, _ in self.args.template]
        inventory_aliases = [alias for _, alias, _, _ in self.args.template]
        inv.create_template_groups_from_alpha_mapping(
            inventory, aliases=inventory_aliases)

        # set up equipment templates
        equipped = [name for name, _, _, equip in self.args.template
                    if int(equip)]
        equipped_aliases = [alias for _, alias, _, equip in self.args.template
                            if int(equip)]
        eq.create_template_groups_from_alpha_mapping(
            equipped, aliases=equipped_aliases)

    def update(self):
        """Update all the containers we care about."""
        self.client.bank.update()
        self.client.tabs.update()
        self.client.chat.update()

    def action(self):
        """No action to take, but log the state of the containers."""

        bank = getattr(self.client.bank, self.args.bank_tab)
        inv = self.client.tabs.inventory
        chat = self.client.chat.all

        for widget in (bank, inv, chat):
            self.msg.append(
                f'{widget.name}: {widget.located}, ({widget.state})')
            self.msg.append(f'i: {widget.interface.located}')

def main():
    app = TestContainersApp(msg_length=200)
    app.setup()
    app.run()


if __name__ == '__main__':
    main()
