import fnmatch
import os
import numpy as np
import chess.pgn


def replace_tags(board):
    board_san = board.split(" ")[0]
    board_san = board_san.replace("2", "11")
    board_san = board_san.replace("3", "111")
    board_san = board_san.replace("4", "1111")
    board_san = board_san.replace("5", "11111")
    board_san = board_san.replace("6", "111111")
    board_san = board_san.replace("7", "1111111")
    board_san = board_san.replace("8", "11111111")
    for i in range(len(board.split(" "))):
        if i > 0 and board.split(" ")[i] != "":
            board_san += " " + board.split(" ")[i]
    return board_san


def find_files(directory, pattern="*.pgn"):
    """Recursively finds all files matching the pattern."""
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))
    return files


def pgn_to_txt(directory):
    """Generator that yields text raw from the directory."""
    files = find_files(directory)
    for filename in files:
        text = ""
        k = 0
        pgn = open(filename)
        offset = 0
        for headers in iter(lambda: chess.pgn.read_headers(pgn), None):
            offset = pgn.tell()
            pgn.seek(offset)
            game = chess.pgn.read_game(pgn)
            node = game
            nm = 1
            while not node.is_end():
                board_fen = replace_tags(node.board().fen())
                next_node = node.variation(0)
                label_san = node.board().san(next_node.move)
                text += board_fen + ":" + label_san + "\n"
                nm += 1
                node = next_node
            if k % 1000 == 0 and k > 1:
                print("Saving file: " + filename + "-" + str(k) + ".txt")
                y = []
                for index, item in enumerate(text):
                    y.append(text[index])
                y = np.array(y)
                np.savetxt(
                    filename + "-" + str(k) + ".txt",
                    y.reshape(1, y.shape[0]),
                    delimiter="",
                    newline="\n",
                    fmt="%s",
                )
                text = ""
            k += 1
        pgn.close()
        yield filename


def pgn_to_label(directory):
    """Generator that yields text raw from the directory."""
    files = find_files(directory)
    text = " "
    for filename in files:
        k = 0
        pgn = open(filename)
        offset = 0
        for headers in chess.pgn.read_headers(pgn):
            offset = pgn.tell()
            pgn.seek(offset)
            game = chess.pgn.read_game(pgn)
            node = game
            while not node.is_end():
                next_node = node.variation(0)
                label_san = node.board().san(next_node.move)
                if " " + label_san + " " not in text:
                    text += label_san + " "
                node = next_node
            if k % 100 == 0 and k > 1:
                print("Labeling file: " + filename + ", step: " + str(k))
            k += 1
        pgn.close()
    y = []
    for index, item in enumerate(text):
        y.append(text[index])
    y = np.array(y)
    np.savetxt(
        "labels.txt", y.reshape(1, y.shape[0]), delimiter="", newline="\n", fmt="%s"
    )


def main():
    iterator = pgn_to_txt("./datasets")
    for filename in iterator:
        print("Done with ", filename)
    pgn_to_label("./datasets")

    print("Done")


if __name__ == "__main__":
    main()
