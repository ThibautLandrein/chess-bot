# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 17:57:15 2022

@author: Thibaut Landrein
"""

from strategies import MinimalEngine


from play import get_ai_move

class Engine(MinimalEngine):

    def search(self, board, time_limit, ponder, draw_offered):
        return get_ai_move(board, 1)