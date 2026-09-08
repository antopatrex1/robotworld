"""Exercise Workbench's actual callbacks through Viser's event dispatchers.

The GUI transport and scene loading are mocked, so these checks need no server,
browser or robot assets. Synchronous callback jobs deliberately run after edits.
"""
import asyncio
from concurrent.futures import Future
from contextlib import nullcontext
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from viser import _gui_api, _gui_handles, _messages
from robot_world.web_app import Workbench


class InputHandle:
    def __init__(self,value,is_button=False):
        self._impl=SimpleNamespace(value=value,is_button=is_button,removed=False,
                                   sync_cb=None,update_cb=[],update_timestamp=0.)

    @property
    def value(self):
        return self._impl.value

    def on_click(self,callback):
        self._impl.update_cb.append(callback)
        return callback


class DeferredExecutor:
    """Model Viser's thread pool with jobs held until a later prompt edit."""
    def __init__(self):
        self.jobs=[]

    def submit(self,callback,event):
        future=Future()
        self.jobs.append((future,lambda:callback(event)))
        return future

    def drain(self):
        for future,callback in self.jobs:
            future.set_result(callback())
        self.jobs.clear()


class PromptSubmissionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        server=MagicMock()
        gui=server.gui
        form=object.__new__(_gui_handles.GuiFormHandle)
        object.__setattr__(form,'_submit_cb',[])
        gui.add_form.return_value=nullcontext(form)
        gui.add_text.return_value=InputHandle('grab the red mug')
        buttons={}

        def add_button(label,**_):
            buttons[label]=InputHandle(False,is_button=True)
            return buttons[label]

        gui.add_button.side_effect=add_button
        with patch('robot_world.web_app.viser.ViserServer',return_value=server),patch.object(Workbench,'load'):
            self.workbench=Workbench()
        self.form=form
        self.button=buttons['Run command']
        self.gui=gui
        self.executor=DeferredExecutor()
        self.api=SimpleNamespace(
            _container_handle_from_uuid={'form':form},
            _gui_input_handle_from_uuid={'prompt':self.workbench.prompt,'run':self.button},
            _resolve_client=lambda _:object(),
            _thread_executor=self.executor,
            _websock_interface=SimpleNamespace(queue_message=lambda _:None),
        )

    async def edit(self,value):
        await _gui_api.GuiApi._handle_gui_updates(self.api,0,
            _messages.GuiUpdateMessage(uuid='prompt',updates={'value':value}))

    async def submit(self,method):
        if method=='enter':
            await _gui_api.GuiApi._handle_gui_form_submit(self.api,0,
                _messages.GuiFormSubmitMessage(uuid='form'))
        else:
            await _gui_api.GuiApi._handle_gui_updates(self.api,0,
                _messages.GuiUpdateMessage(uuid='run',updates={'value':True}))

    def test_enter_and_run_share_an_async_callback_on_single_line_input(self):
        self.assertFalse(self.gui.add_text.call_args.kwargs['multiline'])
        self.assertEqual(len(self.form._submit_cb),1)
        self.assertEqual(len(self.button._impl.update_cb),1)
        callback=self.form._submit_cb[0]
        self.assertIs(callback,self.button._impl.update_cb[0])
        self.assertTrue(asyncio.iscoroutinefunction(callback))

    async def test_enter_captures_submitted_text_before_a_later_edit(self):
        await self.edit('Can you grab the red mug?')
        await self.submit('enter')
        await self.edit('stop')
        self.executor.drain()
        self.assertEqual(self.workbench.commands.get_nowait(),('prompt','Can you grab the red mug?'))
        self.assertTrue(self.workbench.commands.empty())

    async def test_run_captures_submitted_text_before_a_later_edit(self):
        await self.edit('open hands')
        await self.submit('run')
        await self.edit('grab the red mug')
        self.executor.drain()
        self.assertEqual(self.workbench.commands.get_nowait(),('prompt','open hands'))
        self.assertTrue(self.workbench.commands.empty())

    async def test_consecutive_submissions_keep_both_snapshots_in_order(self):
        for method,text in [('enter','grab the red mug'),('run','stop'),('run','open hands')]:
            await self.edit(text)
            await self.submit(method)
        await self.edit('unsubmitted text')
        self.executor.drain()
        self.assertEqual([self.workbench.commands.get_nowait() for _ in range(3)],
                         [('prompt','grab the red mug'),('prompt','stop'),('prompt','open hands')])
        self.assertTrue(self.workbench.commands.empty())


if __name__=='__main__': unittest.main()
