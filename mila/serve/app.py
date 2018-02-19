"""A simple API server to handle prediction requests."""
import logging
import os

from sanic import Sanic, response
from jinja2 import Environment, FileSystemLoader

from .. import config, predict


env = Environment(loader=FileSystemLoader(
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 'templates')))
app = Sanic()


def render(template, **kwargs):
    return env.get_template(template).render(**kwargs)


@app.route('/model')
async def model_root(request):
    models = os.listdir(config.OUTPUT_DIRECTORY)
    return response.json({'models': models})


@app.route('/model/<model_name>')
async def model_info(request, model_name):
    model_info = predict.model_info(os.path.join(config.OUTPUT_DIRECTORY, model_name))
    if not model_info:
        return response.json({
            'error': 'Model not found: {}'.format(model_name)
        }, status=404)

    if request.headers.get('accept') == 'application/json':
        return response.json(model_info)

    # Yeah, let's happily mix content types :-)
    return response.html(render('index.html', **model_info))


@app.route('/model/<model_name>/prediction', methods=['POST'])
async def model_prediction(request, model_name):
    predictions = predict.predict(request.body, os.path.join(config.OUTPUT_DIRECTORY, model_name), cache_model=True)
    return response.json(predictions)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    app.run(port=config.PORT)