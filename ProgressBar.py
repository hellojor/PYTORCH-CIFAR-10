import time
import sys

TOTAL_BAR_LEN = 20
start_time = time.time()

def ProgressBar(current, total, msg=''):
    global start_time

    if current == 0:
        start_time = time.time()
    
    cur_len = (current*TOTAL_BAR_LEN)//total
    left_len = TOTAL_BAR_LEN - cur_len - 1
    
    sys.stdout.write('[')
    for i in range(cur_len):
        sys.stdout.write('x')
    for i in range(left_len):
        sys.stdout.write(' ')
    sys.stdout.write('] ')

    sys.stdout.write('step: ')
    sys.stdout.write('{:>4}/{:>4}|'.format(current+1, total))

    now_time = time.time()
    step_time = now_time - start_time
    time_status = '{:^5}'.format(check_time_format(step_time))
    sys.stdout.write(' time:' + time_status)

    sys.stdout.write('| ' + '{:20}'.format(msg))

    if current == total - 1:
        sys.stdout.write('\n')
    else:
        sys.stdout.write('\r')
    sys.stdout.flush()

def check_time_format(second : int):
    out = ''

    hour = int(second // 3600)
    if hour != 0:
        out = out + str(hour) + 'h'
    second %= 3600
    min = int(second // 60)
    if min != 0:
        out = out + str(min) + 'm'
    second = int(second % 60)
    if second != 0:
        out = out + str(second) + 's'
    return out