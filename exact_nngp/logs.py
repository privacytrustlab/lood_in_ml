def double_print(text, output_file = None, end = '\n'):
    print(text, end = end)
    if not output_file is None:
        output_file.write(str(text) + end)
        output_file.flush()