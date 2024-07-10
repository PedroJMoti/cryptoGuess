# recomendacion.spec

import os
from PyInstaller.utils.hooks import collect_submodules

block_cipher = None

a = Analysis(['recomendacion.py'],
             pathex=[os.getcwd()],
             binaries=[],
             datas=[],
             hiddenimports=collect_submodules('ta'),
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
          cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name='recomendacion',
          debug=False,
          strip=False,
          upx=True,
          console=False )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               upx_exclude=[],
               name='recomendacion')
