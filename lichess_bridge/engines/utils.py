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
import time

from multiprocessing import Pool, Manager, Lock
#You need to download the stockfish engine online in order to use it to get an
#evaluation of our chess board
ENGINE_PATH = os.sep.join([os.getcwd(), "stockfish", "stockfish_14.1_win_x64_avx2.exe"])

SQUARE_INDEX = {
  'a': 0,
  'b': 1,
  'c': 2,
  'd': 3,
  'e': 4,
  'f': 5,
  'g': 6,
  'h': 7
}
LEN = 1000
def random_board(max_depth=200):
  """this function will create our x (the chess board board)"""
  board = chess.Board()
  depth = random.randrange(0, max_depth)

  for _ in range(depth):
    all_moves = list(board.legal_moves)
    random_move = random.choice(all_moves)
    board.push(random_move)
    if board.is_game_over():
      break

  return board

def stockfish(board, depth):
  with chess.engine.SimpleEngine.popen_uci(ENGINE_PATH) as sf:
    result = sf.analyse(board, chess.engine.Limit(depth=depth))
    score = result['score'].white().score()
    return score


# example: h3 -> 17
def square_to_index(square):
  letter = chess.square_name(square)
  return 8 - int(letter[1]), SQUARE_INDEX[letter[0]]


def split_dims(board):
  # this is the 3d matrix
  board3d = numpy.zeros((14, 8, 8), dtype=numpy.int8)

  # here we add the pieces's view on the matrix
  for piece in chess.PIECE_TYPES:
    for square in board.pieces(piece, chess.WHITE):
      idx = numpy.unravel_index(square, (8, 8))
      board3d[piece - 1][7 - idx[0]][idx[1]] = 1
    for square in board.pieces(piece, chess.BLACK):
      idx = numpy.unravel_index(square, (8, 8))
      board3d[piece + 5][7 - idx[0]][idx[1]] = 1

  # add attacks and valid moves too
  # so the network knows what is being attacked
  aux = board.turn
  board.turn = chess.WHITE
  for move in board.legal_moves:
      i, j = square_to_index(move.to_square)
      board3d[12][i][j] = 1
  board.turn = chess.BLACK
  for move in board.legal_moves:
      i, j = square_to_index(move.to_square)
      board3d[13][i][j] = 1
  board.turn = aux

  return board3d

def fill_array(_val):
  board = random_board()
  b = split_dims(board)
  v = stockfish(board, 0)
  while v is None:
      board = random_board()
      b = split_dims(board)
      v = stockfish(board, 0)
  return b, v

if __name__ == "__main__":
  start_time = time.time()
  pool = Pool(processes=os.cpu_count() - 2)
  b_array = numpy.zeros((LEN, 14, 8, 8), dtype=numpy.int8)
  v_array = numpy.zeros(LEN, dtype=numpy.int16)
  result = pool.map(fill_array, range(LEN))
  pool.close()
  pool.join()
  print(time.time() - start_time)

  for idx, line in enumerate(result):
    arr, grade = line
    b_array[idx] = arr
    v_array[idx] = grade

  numpy.savez_compressed("dataset.npz", b=b_array, v=v_array)
  print("Done")
