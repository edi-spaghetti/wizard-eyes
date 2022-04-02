from unittest import TestCase, mock

from wizard_eyes.game_objects.game_objects import GameObject


class TestGameObject(TestCase):

    def setUp(self):

        patcher = mock.patch(
            'wizard_eyes.game_objects.game_objects.Timeout')
        self.mock_timeout = patcher.start()
        self.addCleanup(patcher.stop)

        self.mock_client_class = mock.MagicMock()
        self.mock_client = self.mock_client_class.return_value

    def test_add_timeout(self):

        type(self.mock_client).time = mock.PropertyMock(return_value=100)

        g = GameObject(self.mock_client, self.mock_client)

        self.assertFalse(g.clicked)
        self.assertListEqual(g._clicked, [])

        g.add_timeout(10)

        self.assertTrue(g.clicked)
        self.assertListEqual(g._clicked, [self.mock_timeout.return_value])

    def test_update_click_timeouts_remove_all(self):
        """
        Test that all timeouts are removed if they are older than current
        client time.
        """

        type(self.mock_client).time = mock.PropertyMock(return_value=100)
        type(self.mock_timeout.return_value).offset = mock.PropertyMock(
            return_value=99.99)

        g = GameObject(self.mock_client, self.mock_client)
        g._clicked = [self.mock_timeout()]

        g.update_click_timeouts()

        self.assertFalse(g.clicked)
        self.assertListEqual(g._clicked, [])

    def test_update_click_timeout_remove_one(self):
        """
        Test timeouts older than client time are removed, but newer ones aren't
        """

        type(self.mock_client).time = mock.PropertyMock(return_value=100)
        type(self.mock_timeout.return_value).offset = mock.PropertyMock(
            return_value=99.99)

        timeout2 = mock.MagicMixin()
        type(timeout2).offset = 100.1

        g = GameObject(self.mock_client, self.mock_client)
        g._clicked = [self.mock_timeout(), timeout2]

        g.update_click_timeouts()

        self.assertTrue(g.clicked)
        self.assertListEqual(g._clicked, [timeout2])
