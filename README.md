# chess-bot
The bot is trained using Tensorflow. To get training data I used the chess python library to get a random chess board and the stockfish engine (which is why stockfish is contained in the repo) to get a grade for the board. The data is stored in a numpy array and saved in a .npz file (see utils.py).

The bridge with lichess is made in lichess_bridge/strategies.py. To play against the bot you need to create a bot account on lichess.org, get a OAuth token from lichess, tune the config.yml file in the lichess_bridge directory (set the protocol to homemade, overload the MinimalEngine class in strategies.py, note that your overloaded class only needs to implement the search method to start playing and type your token in the token field). To start the bot, execute the file lichess_bot.py in lichess_bridge directory, then you're good to go !

To play againt my bot, search for thibaut_bot on lichess.org and challenge it !
