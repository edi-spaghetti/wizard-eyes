from unittest import TestCase, mock

from wizard_eyes.client import Client


class TestMouseOptions(TestCase):

    def setUp(self):
        self.mock_logger()

    def mock_argv(self, *args):

        argv = list(args)
        argv.insert(0, 'test.py')

        patcher = mock.patch('sys.argv', argv)
        self.mock_argv = patcher.start()
        self.addCleanup(patcher.stop)

    def mock_logger(self):
        patcher = mock.patch(
            'wizard_eyes.game_objects.game_objects.logging')
        self.logging = patcher.start()
        self.logger = self.logging.getLogger.return_value
        self.addCleanup(patcher.stop)

    def test_load_templates_walk_here(self):
        """Check templates exists for 'Walk here' per letter."""

        c = Client('RuneLite')
        c.mouse_options.load_templates(set('Walk here'))

        self.assertEqual(len(c.mouse_options.templates), 7)
