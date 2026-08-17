from __future__ import annotations

import argparse
import json
import lzma
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    args = parser.parse_args()

    path = Path(args.file)
    with lzma.open(path, "rt", encoding="utf-8") as fh:
        demo = json.load(fh)

    print("top keys:", sorted(demo.keys()))
    rounds = demo.get("gameRounds", [])
    print("round count:", len(rounds))
    if not rounds:
        return

    rnd = rounds[0]
    print("round keys:", sorted(rnd.keys()))
    for key in ["kills", "frames"]:
        values = rnd.get(key, [])
        print(f"{key} count:", len(values))
        if values:
            print(f"{key} first keys:", sorted(values[0].keys()))
            if key == "frames":
                frame = values[0]
                print("frame t keys:", sorted(frame.get("t", {}).keys()))
                print("frame ct keys:", sorted(frame.get("ct", {}).keys()))
                players = frame.get("t", {}).get("players", [])
                if players:
                    print("player keys:", sorted(players[0].keys()))


if __name__ == "__main__":
    main()

