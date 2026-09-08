import json
import tempfile
from pathlib import Path
import unittest
from robot_world.scene import available_environments


class EnvironmentTests(unittest.TestCase):
    def test_missing_optional_world_is_not_offered_until_its_manifest_is_installed(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            configs=root/'configs/environments';configs.mkdir(parents=True)
            (configs/'lab.json').write_text(json.dumps({'name':'Lab'}))
            (configs/'room.json').write_text(json.dumps({'name':'Room','world_manifest':'data/room.json'}))
            self.assertEqual(available_environments(root),{'Lab':'lab'})
            (root/'data').mkdir();(root/'data/room.json').write_text('{}')
            self.assertEqual(available_environments(root),{'Lab':'lab','Room':'room'})


if __name__=='__main__':unittest.main()
