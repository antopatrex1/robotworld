import unittest

from robot_world.commands import Command, parse_command
from robot_world.objects import OBJECTS


class CommandTests(unittest.TestCase):
    def test_grab_red_mug_accepts_normal_request_wording(self):
        for text in (
            'grab the red mug',
            'Can you grab the red mug?',
            'could you please grab the red mug, please!',
            'Robot, please grasp the red mug.',
            'pick the red mug up',
            'grab a red mug',
            '\n  GRAB   the\tred mug  \n',
        ):
            with self.subTest(text=text):
                self.assertEqual(parse_command(text),Command('pick',('mug',)))

    def test_every_labeled_object_maps_to_its_own_grasp_target(self):
        for key,info in OBJECTS.items():
            with self.subTest(object=key):
                self.assertEqual(parse_command('please grab the '+info['label']),Command('pick',(key,)))

    def test_green_apple_pick_requests_resolve_to_apple_not_previous_object(self):
        for text in (
            'pick up the green apple',
            'Pick up the apple.',
            'grab the green apple',
            'Can you grab the green apple?',
            'could you please pick the green apple up',
            'grasp an apple, please',
            '\n  grab   the GREEN APPLE \t',
        ):
            with self.subTest(text=text):
                self.assertEqual(parse_command(text),Command('pick',('apple',)))

    def test_wrappers_preserve_existing_controls(self):
        self.assertEqual(parse_command('Can you walk backward 1 meter?'),Command('walk',('backward',1.)))
        self.assertEqual(parse_command('please walk to x -1.5 y 0.8.'),Command('position',(-1.5,.8)))
        self.assertEqual(parse_command('Open hands, please!'),Command('hands',(0.,)))
        self.assertEqual(parse_command('Could you stop?'),Command('stop'))

    def test_unknown_objects_compound_actions_and_unbounded_values_are_rejected(self):
        for text in (
            'please grab the red mug and open hands',
            'grab the red mug\nstop',
            'can you grab the blue mug',
            'grab the red apple',
            'grab the red mug twice',
            'can you run shell commands',
            'please walk forward 500 meters',
            'walk forward nan meters',
        ):
            with self.subTest(text=text),self.assertRaises(ValueError):
                parse_command(text)

    def test_empty_submission_explains_how_to_submit(self):
        for text in ('',' \n\t ','please'):
            with self.subTest(text=text),self.assertRaisesRegex(ValueError,'Type a command'):
                parse_command(text)


if __name__=='__main__': unittest.main()
