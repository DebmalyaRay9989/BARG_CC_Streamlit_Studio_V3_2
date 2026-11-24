"""Streamlit AppTest including the interactive document inspector regression."""
from streamlit.testing.v1 import AppTest
from pathlib import Path
import os
import logging

class LegacyWidthLogCapture(logging.Handler):
    def __init__(self):
        super().__init__(logging.WARNING)
        self.messages = []

    def emit(self, record):
        message = record.getMessage()
        if "use_container_width" in message or "use_column_width" in message:
            self.messages.append(message)

width_log = LegacyWidthLogCapture()
# Streamlit loggers can have propagation disabled; attach to existing and future
# parent loggers, as well as the root logger.
logging.getLogger().addHandler(width_log)
logging.getLogger("streamlit").addHandler(width_log)
for name, logger in list(logging.Logger.manager.loggerDict.items()):
    if name.startswith("streamlit.") and isinstance(logger, logging.Logger):
        logger.addHandler(width_log)
os.chdir(Path(__file__).resolve().parent)
a=AppTest.from_file('streamlit_app.py',default_timeout=120).run()
assert not a.exception, a.exception

def check(label):
    assert not a.exception, (label,a.exception)
    assert not a.error, (label,[e.value for e in a.error])
    # Inspect user-facing text elements, not widget internals or uploaded document data.
    for kind in ('markdown', 'text', 'caption', 'code'):
        for element in a.get(kind):
            value = getattr(element, 'value', '')
            if isinstance(value, str):
                assert 'Creator of Delta protobuf messages' not in value, (label, kind, value)
                assert 'DeltaGenerator(_provided_cursor=' not in value, (label, kind, value)
    print('PASS',label)

def button(label):
    return next(w for w in a.button if w.label==label)

appearance=next(w for w in a.sidebar.selectbox if w.label=='Appearance')
for option in appearance.options:
    next(w for w in a.sidebar.selectbox if w.label=='Appearance').set_value(option).run()
    check('appearance: '+option)
pages=a.sidebar.radio[0].options
for p in pages:
    a.sidebar.radio[0].set_value(p).run()
    check(p)
    if p.startswith('02'):
        next(w for w in a.multiselect if w.label.startswith('Remove available')).set_value([0]).run()
        check('selected view-loss curve')
    if p.startswith('05'):
        button('Load bundled mixed-document demo archive').click().run()
        check('demo import')
        next(w for w in a.radio if w.label=='Computation profile').set_value('Quick').run()
        button('Run clustering and matched baseline').click().run()
        check('clustering and first document inspector')
        assert any('keywords' in getattr(e, 'value', '').lower() for e in a.subheader)
        assert not a.exception
        inspector=next(w for w in a.selectbox if w.label=='Individual document inspector')
        assert len(inspector.options)>1
        inspector.select_index(1).run()
        check('second document inspector')
        button('Audit one-refit stability').click().run()
        check('one-refit audit')
    if p.startswith('06'):
        button('Execute new view-loss experiment').click().run()
        check('live Digits simulation')
    if p.startswith('07'):
        button('Verify bundled evidence hashes').click().run()
        check('evidence integrity status')
        assert a.success
assert not width_log.messages, ('Deprecated width API warnings', width_log.messages)
print('Workspace and interaction checks passed; no captured legacy width warnings. Browser rendering still requires manual check.')
