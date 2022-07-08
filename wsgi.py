from sys import implementation
from app.main import app

# start the server with the 'run()' method
if __name__ == '__main__':
    app.run(debug=True)