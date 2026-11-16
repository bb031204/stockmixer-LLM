import ast
import contextlib
import io
import os
import re
import shutil
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from news_fusion import console,experiment
from news_fusion.outputs import Tee,verify_bundle
from tests_news.test_workflow import temporary_project,setup_project,ToyModel


class Tty(io.StringIO):
    @property
    def encoding(self):return 'utf-8'
    def isatty(self):return True


def reference_console():
    path=Path('D:/finance/model/StockMixer/StockMixer/src/train.py')
    source=ast.parse(path.read_text(encoding='utf-8-sig'))
    names={'SimpleProgress','progress_iter','format_duration','format_number','format_metric','format_loss','color_text','compact_epoch_summary'}
    nodes=[n for n in source.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    scope={'np':np,'time':time,'shutil':shutil,'sys':sys}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'reference_console','exec'),scope)
    return scope


class ConsoleTests(unittest.TestCase):
    def test_decimal_numbers_preserve_tiny_nonzero_values_without_exponents(self):
        for value in (1e-18,-1e-9,6.71e-2,1e20):
            text=console.format_number(value)
            self.assertNotRegex(text,r'[eE]')
            self.assertNotEqual(float(text),0)
            for formatter in (console.format_loss,console.format_metric):
                self.assertNotRegex(formatter(value),r'[eE]')
        for value in (None,float('nan'),float('inf')):
            self.assertEqual(console.format_number(value),'N/A')
        self.assertEqual(console.format_number(-0.0),'0')
        self.assertEqual(console.format_metric(0.000018919587091659196),'.0000189')

    def test_compact_epoch_line_matches_reference_project_exactly(self):
        ref=reference_console()
        values=(1,100,.0671,.0129,{'IC':.0273,'RIC':.223,'prec_10':.519,'sharpe5':2.44},
            {'IC':.00752,'RIC':.103,'sharpe5':1.43},27,2673,1,'val_loss',.0129)
        with patch('sys.__stdout__',io.StringIO()):
            actual=console.compact_epoch_summary(*values)
            expected=ref['compact_epoch_summary'](*values)
        self.assertEqual(actual,expected)
        for token in ('BEST MODEL','Epoch=001/100','TrainLoss=0.0671','ValLoss=0.0129','ValIC=.027','TestIC=.008','ETA=44m33s'):
            self.assertIn(token,actual)

    def test_tty_progress_bar_matches_reference_and_overwrites_one_line(self):
        ref=reference_console()
        def render(cls):
            stream=Tty()
            with patch('sys.__stdout__',stream),patch('shutil.get_terminal_size',return_value=os.terminal_size((110,25))),patch('time.time',return_value=10.):
                bar=cls(range(2),total=2,desc='Epoch 001 Train',log_every=1)
                for _ in bar:bar.set_postfix(loss='0.0671',rank='0.00138')
            return stream.getvalue()
        actual=render(console.SimpleProgress)
        self.assertEqual(actual,render(ref['SimpleProgress']))
        self.assertIn('100.0%',actual)
        self.assertIn('2/2',actual)
        self.assertIn('\r\x1b[2K',actual)
        self.assertNotIn('e-',actual)

    def test_non_tty_output_does_not_emit_bar_escape_codes(self):
        stream=io.StringIO()
        with patch('sys.__stdout__',stream):
            list(console.progress_iter(range(3),total=3,desc='Epoch 001 Train'))
        self.assertEqual(stream.getvalue(),'')

    def test_nested_logs_strip_color_but_real_tty_keeps_best_highlight(self):
        terminal=Tty();workflow_log=io.StringIO();train_log=io.StringIO()
        tee=Tee(Tee(terminal,workflow_log),train_log)
        tee.write('\x1b[92mBEST MODEL\x1b[0m')
        self.assertIn('\x1b[92m',terminal.getvalue())
        self.assertEqual(train_log.getvalue(),'BEST MODEL')
        self.assertEqual(workflow_log.getvalue(),'BEST MODEL')

    def test_training_has_reference_sections_summary_footer_and_no_scientific_numbers(self):
        with temporary_project() as tmp:
            root=Path(tmp);config=setup_project(root)
            output=io.StringIO()
            with patch.object(experiment,'make_model',side_effect=lambda *a:ToyModel()),contextlib.redirect_stdout(output):
                result=experiment.train(root,config,root/'outputs/data',root/'outputs/console_run',epochs=2,smoke=True)
            text=output.getvalue()
            for token in ('StockMixer Training','[RUN   ]','[ENV   ]','[SEED  ]','[DATA  ]','[SPLIT ]','[WINDOW]',
                          '[TRAIN ]','[SCALES]','[PARAM ]','Epoch=001/2','Epoch=002/2','Final Test','Metrics     IC=','[OUTPUT]','[FILE  ]'):
                self.assertIn(token,text)
            self.assertNotRegex(text,r'(?<![\w/\\])[-+]?\d+(?:\.\d+)?[eE][+-]?\d+(?!\w)')
            self.assertNotIn('################',text)
            self.assertEqual(result['test_evaluations'],2)
            self.assertNotIn('\x1b[',(root/'outputs/console_run/logs/train.log').read_text())
            verify_bundle(root/'outputs/console_run')


if __name__=='__main__':unittest.main()
