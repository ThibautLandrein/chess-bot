# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 15:30:31 2022

@author: Thibaut Landrein
"""

import chess
import chess.engine
import numpy
import tflite_runtime.interpreter as tflite
from utils import split_dims, ENGINE_PATH


model = tflite.Interpreter("/home/pi/chess/chess-bot/model.tflite")
def minimax_eval(board):
  board3d = split_dims(board)
 # board3d = numpy.expand_dims(board3d, 0)
  input_details = model.get_input_details()
  input_data = numpy.array(board3d, dtype=numpy.float32)
  output_details = model.get_output_details()
  model.allocate_tensors()
  model.set_tensor(input_details[0]['index'], [input_data])
  model.invoke()
  output_data = model.get_tensor(output_details[0]['index'])
  return output_data

def minimax(board, depth, alpha, beta, maximizing_player):
  if depth == 0 or board.is_game_over():
    return minimax_eval(board)

  if maximizing_player:
    max_eval = -numpy.inf
    for move in board.legal_moves:
      board.push(move)
      eval = minimax(board, depth - 1, alpha, beta, False)
      board.pop()
      max_eval = max(max_eval, eval)
      alpha = max(alpha, eval)
      if beta <= alpha:
        break
    return max_eval
  else:
    min_eval = numpy.inf
    for move in board.legal_moves:
      board.push(move)
      eval = minimax(board, depth - 1, alpha, beta, True)
      board.pop()
      min_eval = min(min_eval, eval)
      beta = min(beta, eval)
      
      if beta <= alpha:
        break
    return min_eval


# this is the actual function that gets the move from the neural network
def get_ai_move(board, depth):
  max_move = None
  max_eval = -numpy.inf

  for move in board.legal_moves:
    board.push(move)
    eval = minimax(board, depth - 1, -numpy.inf, numpy.inf, False)
    board.pop()
    if eval > max_eval:
      max_eval = eval
      max_move = move
  return max_move
