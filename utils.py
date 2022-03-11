# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 20:07:12 2022

@author: Thibaut Landrein
"""

import chess
import chess.engine
import random
import numpy
import os



ENGINE_PATH = os.sep.join([os.getcwd(), "stockfish", "stockfish_14.1_win_x64_avx2.exe"])
# this function will create our x (board)
def random_board(max_depth=200):
  board = chess.Board()
  depth = random.randrange(0, max_depth)

  for _ in range(depth):
    all_moves = list(board.legal_moves)
    random_move = random.choice(all_moves)
    board.push(random_move)
    if board.is_game_over():
      break

  return board


# this function will create our f(x) (score)
def stockfish(board, depth):
  with chess.engine.SimpleEngine.popen_uci(ENGINE_PATH) as sf:
    result = sf.analyse(board, chess.engine.Limit(depth=depth))
    score = result['score'].white().score()
    return score

if __name__ == "__main__":
    board = random_board()
    print(stockfish(board, 10))