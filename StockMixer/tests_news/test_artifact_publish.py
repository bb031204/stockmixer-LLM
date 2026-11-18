import os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from news_fusion.io import new_artifact

class Tests(unittest.TestCase):
    def test_transient_windows_handle_does_not_lose_valid_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            target=Path(tmp)/'ready';real=os.rename;calls=[]
            def rename(source,destination):
                calls.append(1)
                if len(calls)==1:
                    error=PermissionError('transient handle');error.winerror=5;raise error
                return real(source,destination)
            with patch('news_fusion.io.os.rename',side_effect=rename),patch('news_fusion.io.time.sleep'):
                with new_artifact(target) as stage:(stage/'evidence.txt').write_text('verified',encoding='utf-8')
            self.assertEqual((target/'evidence.txt').read_text(),'verified');self.assertEqual(len(calls),2)
    def test_existing_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            target=Path(tmp)/'ready'
            with self.assertRaises(FileExistsError):
                with new_artifact(target) as stage:
                    (stage/'evidence.txt').write_text('new');target.mkdir();(target/'original.txt').write_text('old')
            self.assertEqual((target/'original.txt').read_text(),'old');self.assertFalse((target/'evidence.txt').exists())
    def test_nontransient_permission_error_is_not_hidden(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch('news_fusion.io.os.rename',side_effect=PermissionError('permanent')),patch('news_fusion.io.time.sleep') as sleep:
                with self.assertRaises(PermissionError):
                    with new_artifact(Path(tmp)/'ready') as stage:(stage/'evidence.txt').write_text('test')
                sleep.assert_not_called()

if __name__=='__main__':unittest.main()
