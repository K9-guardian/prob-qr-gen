#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, 'python_qrcode')
import qrcode
import urllib
from urllib.parse import urlparse, urlunparse
import argparse

CHARS = "aeioubmcnsvdfghjklpqrtwxyz0123456789"
# PROB_BYTE_RANGE = ((0, 5), (6, 10), (11, 16))
# PROB_BYTE_RANGE = [(0, 16)]
PROB_BYTES = (1, 2, 3)
LUMINANCES = (
    ((0, 0, 0), (255, 255, 255)),
    ((0, 0, 0), (178, 178, 178)),
    ((0, 0, 0), (100, 100, 100))
)

def gen_qrs(url: urllib.parse.ParseResult):
    """Generates 10 unique probabilistic QRs for a given input URL"""
    success = False
    domain_list = url.netloc.split(".")

    tld = domain_list[-1]
    if len(domain_list) > 2:
        name = domain_list[1:-1]
        sub = domain_list[0]
    else:
        sub = ""
        name = domain_list[0]

    for i in range(len(name)-1, 0, -1):
        if success:
            break
        for ch in CHARS:
            if success:
                break
            if ch == name[i]:
                continue
            prob_bits = 0
            for byte in PROB_BYTES:
                qr = qrcode.QRCode(
                    error_correction=qrcode.constants.ERROR_CORRECT_M,
                    box_size=40,
                    prob_bytes=byte
                )
                qr.add_data(urlunparse(url), real=True)
                new_name = name[:i]+ch+name[i+1:]
                new_url = f"{url.scheme}://{sub}{new_name}.{tld}"
                qr.add_data(new_url, real=False)
                mat = qr.get_matrix()
                prob_bits += sum(1 for row in mat for elem in row if isinstance(elem, dict))
                qr.clear()
            if prob_bits <= 6:
                success = True
                print(new_url)
                for byte in PROB_BYTES:
                    for i, lum in enumerate(LUMINANCES):
                        qr = qrcode.QRCode(
                            error_correction=qrcode.constants.ERROR_CORRECT_M,
                            box_size=40,
                            prob_bytes=byte
                        )
                        qr.add_data(urlunparse(url), real=True)
                        qr.add_data(new_url, real=False)
                        img = qr.make_image(fill_color=lum[0], back_color=lum[1])
                        img.save(f"./dataset/{new_name}-{byte}-lum{i}.png")
                        qr.clear()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-u", "--url")
    parser.add_argument("-f", "--file")
    args = parser.parse_args()
    parsed_url = urlparse(args.url)
    os.makedirs("./dataset/", exist_ok=True)

    if (args.file):
        with open(args.file, "r") as f:
            for line in f:
                gen_qrs(urlparse(line.strip()))
    else:
        gen_qrs(parsed_url, "qr.png")

if __name__ == "__main__":
    main()
