# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from app import App
import slangpy as spy
import sgl
from time import time

# Create app
app = App()

# Load shaders
examplemodule = spy.Module.load_from_file(app.device, "example.slang")

# Record start time
start = time()

# Reset time when keyboard hit


def kb_event(e: sgl.KeyboardEvent):
    global start
    if e.key == sgl.KeyCode.space and e.type == sgl.KeyboardEventType.key_press:
        start = time()


app.on_keyboard_event = kb_event

# Run app
while app.process_events():
    examplemodule.sinwave(t=time()-start, _result=app.output)
    app.present()
