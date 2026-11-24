"""Document ingestion and exploratory BARG-CC clustering.
No semantic claims without independent labels; K selected without test labels.
"""
import base64, hashlib, io, json, re, zipfile
from pathlib import PurePosixPath
import numpy as np
import pandas as pd

MAX_ARCHIVE=500_000_000
MAX_FILES=1500
MAX_ITEM=12_000_000
ALLOWED={'.png','.jpg','.jpeg','.tif','.tiff','.bmp','.webp','.txt','.md','.pdf'}

def extract_archive(contents, filename):
    if not filename or not filename.lower().endswith('.zip'): raise ValueError('Upload a ZIP of image, text, or PDF documents.')
    raw=base64.b64decode(contents.partition(',')[2],validate=True)
    if len(raw)>MAX_ARCHIVE: raise ValueError('Archive exceeds 500 MB upload limit.')
    return _extract_zip(io.BytesIO(raw))

def extract_archive_file(path):
    """Load a local ZIP without sending its bytes through the browser."""
    from pathlib import Path
    path=Path(path)
    if path.suffix.lower()!='.zip':raise ValueError('Select a .zip file.')
    if path.stat().st_size>500_000_000:raise ValueError('ZIP exceeds the 500 MB local-file limit.')
    return _extract_zip(path)

def _extract_zip(source, ocr=True, progress=None):
    records=[]; seen={}; skipped=[]
    with zipfile.ZipFile(source) as z:
        files=[i for i in z.infolist() if not i.is_dir() and not i.filename.startswith('__MACOSX/')]
        if len(files)>MAX_FILES: raise ValueError('Maximum 1500 files per archive.')
        if sum(i.file_size for i in files)>180_000_000: raise ValueError('Uncompressed archive exceeds 180 MB.')
        for index,info in enumerate(files):
            if progress: progress(index,len(files),f'Reading {info.filename[:80]}')
            path=PurePosixPath(info.filename)
            if '..' in path.parts or path.is_absolute() or '\\' in info.filename or info.file_size>MAX_ITEM or info.flag_bits & 1:
                skipped.append({'file':info.filename,'reason':'unsafe, encrypted, or oversized'});continue
            suffix=path.suffix.lower()
            if suffix not in ALLOWED:
                skipped.append({'file':info.filename,'reason':'unsupported format'});continue
            blob=z.read(info)
            digest=hashlib.sha256(blob).hexdigest()
            if digest in seen:
                skipped.append({'file':info.filename,'reason':'exact duplicate of '+seen[digest]});continue
            seen[digest]=info.filename
            rec={'document_id':digest[:16],'filename':info.filename,'type':suffix,'size_bytes':len(blob),'sha256':digest,'text':'','image_features':None,'ocr_status':'not_applicable'}
            try:
                if suffix in ('.txt','.md'):
                    rec['text']=blob.decode('utf-8-sig',errors='replace')[:100000]
                elif suffix=='.pdf':
                    try:
                        from pypdf import PdfReader
                        reader=PdfReader(io.BytesIO(blob));rec['text']='\n'.join(p.extract_text() or '' for p in reader.pages[:30])[:100000]
                        rec['ocr_status']='pdf_text_extraction_only'
                    except ImportError: rec['ocr_status']='pypdf_not_installed'
                else:
                    from PIL import Image,ImageOps,ImageStat,ImageFilter
                    im=Image.open(io.BytesIO(blob))
                    if im.width*im.height>32_000_000: raise ValueError('image pixel limit exceeded')
                    im.load()
                    gray=ImageOps.grayscale(im)
                    thumb=ImageOps.pad(gray,(128,128),color=255)
                    a=np.asarray(thumb,dtype=np.float32)/255.
                    ink=1-a
                    row=ink.mean(axis=1).reshape(16,8).mean(axis=1)
                    col=ink.mean(axis=0).reshape(16,8).mean(axis=1)
                    gx=np.abs(np.diff(a,axis=1));gy=np.abs(np.diff(a,axis=0))
                    rec['image_features']=np.r_[row,col,[ink.mean(),ink.std(),np.quantile(ink,.9),gx.mean(),gy.mean(),np.mean(ink>.5),np.log1p(im.width),np.log1p(im.height),im.width/max(im.height,1)]].astype(float).tolist()
                    rec['width']=im.width;rec['height']=im.height
                    if not ocr:
                        rec['ocr_status']='disabled_by_user'
                        records.append(rec);continue
                    try:
                        import pytesseract
                        rec['text']=pytesseract.image_to_string(im,config='--psm 3',timeout=8)[:100000]
                        rec['ocr_status']='completed'
                    except ImportError: rec['ocr_status']='pytesseract_not_installed'
                    except Exception as e: rec['ocr_status']='ocr_unavailable_or_failed'
            except Exception as e:
                skipped.append({'file':info.filename,'reason':str(e)[:140]});continue
            records.append(rec)
    if progress: progress(len(files),len(files),'Import complete')
    return records,skipped

