import unittest

from wizard_eyes.log_overlay import Lines, tidy


class TestLines(unittest.TestCase):

    def test_tidy(self):
        self.assertEqual(
            tidy('Time 0h 1m 2s 0.551 - world hop in: 2.9 hours - doing X'),
            '0h 1m 2s  doing X')
        self.assertIsNone(tidy('Time 0h 1m 2s 0.5 - world hop in: 2.9 hours'
                               ' - AFK: 0.320'))
        self.assertEqual(tidy('Time 1h 1m 2s 0.551 - waiting: 4.99'),
                         '1h 1m 2s  waiting')
        self.assertEqual(tidy('  Collect:   done '), 'Collect: done')
        self.assertIsNone(tidy(''))

    def test_only_different_lines(self):
        lines = Lines(3)
        for i in range(4):
            lines.add(f'Time 0h 0m {i}s 0.5 - waiting')
        lines.add('a thing happened')
        lines.add('Time 0h 0m 9s 0.5 - working')
        self.assertEqual(list(lines.lines), [
            '0h 0m 3s  waiting', 'a thing happened', '0h 0m 9s  working'])
        lines.add('another')
        self.assertEqual(len(lines.lines), 3)
        self.assertEqual(lines.text(4).splitlines()[-1], 'anot')


if __name__ == '__main__':
    unittest.main()
