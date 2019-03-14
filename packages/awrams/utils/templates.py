
def transform_file(infile,outfile,templates):
    with open(infile,'r') as fh:
        indata = fh.read()
    lines = indata.split('\n')

    outlines = []

    for line in lines:
        if '//ATL_BEGIN' in line:
            start = line.find('//ATL_BEGIN')
            spacing = line[:start]
            start = line.find('<') + 1
            end = line.find('>')
            tkey = line[start:end]
            ttxt = templates[tkey]
            for tl in ttxt:
                outlines.append(spacing + tl)
        else:
            outlines.append(line)

    with open(outfile,'w') as fh:
        for line in outlines:
            fh.write(line+'\n')