import json
from quart import Quart, request

from python_api import model_functions

app = Quart(__name__)


@app.route('/suggestions', methods=['POST'])
async def suggestions():
    number = int(request.args.get('number', 10))
    body = await request.data
    data = json.loads(body)
    print('================================')
    print(json.dumps(data, indent=2))
    print('================================')

    suggestions = model_functions.suggest_songs(data, number)

    return json.dumps(suggestions), 200


app.run(port=2000)
