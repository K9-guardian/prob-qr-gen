Computer security project for UCSD CSE227 (WI25).

Usage:
```py
import sys
sys.path.insert(0, 'python_qrcode')
import qrcode

qr = qrcode.QRCode(
    border=2,
    error_correction=qrcode.constants.ERROR_CORRECT_M,
    box_size=40,
    prob_bytes=1, # modified
    debug=True # modified
)
qr.add_data("https://kumarde.com", real=True) # modified
qr.add_data("https://kunarde.com", real=False) # modified
img = qr.make_image(debug=True)
img.save("prob_qr.png")
```

The parameters mostly match that of [`python-qrcode`](https://pypi.org/project/qrcode/).
The new parameters are `prob_bytes`, which controls the number of probabilistic bytes, and `real` in `add_data`, which tells the generator which URL is real and fake.
These URLs should be 1 character apart to simulate a real [Qishing](https://www.cloudflare.com/learning/security/what-is-quishing/) attack.
Deterministic changes to the EC segment are chosen randomly as of now, but future work should look into making these changes burst errors to test scanner ability to correct them.
