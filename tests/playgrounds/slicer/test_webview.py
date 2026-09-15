import webview
import traceback

def test_create_window():
    def verify(window):
            #window.events.loaded.wait(timeout=5)
            content = window.evaluate_js('document.body.innerText')
            result['text']= content
            window.destroy()

    window = webview.create_window('Test', html='<html><body>Hola, mundo</body></html>')
    result = {}
    webview.start(verify, window)

    assert result['text'] == 'Hola, mundo'