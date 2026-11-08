"""The Lichess puzzle database (CC0, database.lichess.org) as move-prediction examples.

Each puzzle is a position, the opponent's move that sets it up, and the solution. Every one of the solver's moves is
an example: the position (mirrored so the side to move is always white), the legal moves, and the solution move.
Puzzles are split by a fixed shuffle into train, validation (20,000) and test (50,000); test and validation keep
rating, themes and the move's place in the solution. Writes .npy arrays and themes.json to LAB_DEST.
python chess_data.py '<json>' prints one JSON line.
"""
import csv, io, json, os, sys, time, urllib.request
from multiprocessing import Pool
from pathlib import Path
import numpy as np, zstandard, chess
DEST = Path(os.environ['LAB_DEST'])
URL = 'https://database.lichess.org/lichess_db_puzzle.csv.zst'


def encode(board):
    sq = np.zeros(64, np.int8)
    for s, p in board.piece_map().items():
        sq[s] = p.piece_type + (0 if p.color == chess.WHITE else 6)
    extra = np.zeros(5, np.int8)
    extra[:4] = [board.has_kingside_castling_rights(chess.WHITE), board.has_queenside_castling_rights(chess.WHITE),
                 board.has_kingside_castling_rights(chess.BLACK), board.has_queenside_castling_rights(chess.BLACK)]
    extra[4] = 0 if board.ep_square is None else 1 + chess.square_file(board.ep_square)
    bits = np.zeros(4096, bool)
    for m in board.legal_moves:
        bits[m.from_square * 64 + m.to_square] = True
    return sq, extra, np.packbits(bits)


def examples(row):
    """All solver moves of one puzzle, from the solver's side."""
    try:
        board = chess.Board(row['FEN'])
        moves = row['Moves'].split()
        board.push_uci(moves[0])
        out = []
        for k in range(1, len(moves)):
            mv = chess.Move.from_uci(moves[k])
            if k % 2 == 1:
                b, m = (board.mirror(), chess.Move(chess.square_mirror(mv.from_square), chess.square_mirror(mv.to_square))) if board.turn == chess.BLACK else (board, mv)
                sq, extra, mask = encode(b)
                label = m.from_square * 64 + m.to_square
                assert np.unpackbits(mask)[label], 'solution is not legal'
                out.append((sq, extra, mask, label, k // 2))
            board.push(mv)
        return out
    except Exception:
        return []


def main(job):
    t0 = time.time()
    raw = urllib.request.urlopen(URL, timeout=600).read()
    text = zstandard.ZstdDecompressor().stream_reader(io.BytesIO(raw)).read().decode()
    rows = list(csv.DictReader(io.StringIO(text)))
    order = np.random.default_rng(0).permutation(len(rows))
    split = {'test': order[:50000], 'val': order[50000:70000], 'train': order[70000:70000 + job.get('train_puzzles', len(rows))]}
    info = dict(url=URL, puzzles=len(rows), download_mb=len(raw) / 1e6, seconds_download=time.time() - t0)
    themes = sorted({t for r in rows for t in r['Themes'].split()})
    with Pool(job.get('procs', 16)) as pool:
        for name, ix in split.items():
            cols = {k: [] for k in ('sq', 'extra', 'mask', 'label', 'ply', 'puzzle')}
            for c in range(0, len(ix), 200000):  # chunks keep memory compact
                part = ix[c:c + 200000]
                got = pool.map(examples, [rows[i] for i in part], chunksize=2000)
                flat = [(p, e) for p, ex in zip(part, got) for e in ex]
                cols['sq'].append(np.stack([e[0] for _, e in flat])); cols['extra'].append(np.stack([e[1] for _, e in flat]))
                cols['mask'].append(np.stack([e[2] for _, e in flat])); cols['label'].append(np.array([e[3] for _, e in flat], np.int16))
                cols['ply'].append(np.array([e[4] for _, e in flat], np.int8)); cols['puzzle'].append(np.array([p for p, _ in flat], np.int32))
                del got, flat
            for k, v in cols.items():
                np.save(DEST / f'{name}_{k}.npy', np.concatenate(v))
            info[f'{name}_examples'] = int(sum(len(v) for v in cols['label']))
            info[f'{name}_puzzles'] = len(ix)
            if name != 'train':
                meta = {int(i): dict(rating=int(rows[i]['Rating']), themes=rows[i]['Themes'].split(), id=rows[i]['PuzzleId']) for i in ix}
                (DEST / f'{name}_meta.json').write_text(json.dumps(meta))
            del cols
    (DEST / 'themes.json').write_text(json.dumps(themes))
    info['seconds'] = time.time() - t0
    (DEST / 'info.json').write_text(json.dumps(info))
    return info


if __name__ == '__main__':
    print(json.dumps(main(json.loads(sys.argv[1]))))
