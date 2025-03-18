#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
sys.path.insert(0, 'python_qrcode')
import qrcode


# In[ ]:


qr = qrcode.QRCode(border=2, error_correction=qrcode.constants.ERROR_CORRECT_M, box_size=40)
qr.add_data("https://kumarde.com")
img = qr.make_image()
img.save("real_qr.png")


# In[ ]:


qr = qrcode.QRCode(border=2, error_correction=qrcode.constants.ERROR_CORRECT_M, box_size=40)
qr.add_data("https://kunarde.com")
img = qr.make_image()
img.save("fake_qr.png")


# In[ ]:


qr = qrcode.QRCode(border=2, error_correction=qrcode.constants.ERROR_CORRECT_M, box_size=40, prob_bytes=1, debug=True)
qr.add_data("https://kumarde.com", real=True)
qr.add_data("https://kunarde.com", real=False)
img = qr.make_image(debug=True)
img.save("prob_qr.png")


# In[ ]:




