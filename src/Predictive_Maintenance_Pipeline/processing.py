CRITICAL = 'CRITICAL'
WARNING = 'WARNING'
NORMAL = 'NORMAL'
CRITICAL_RUL = 30
WARNING_RUL = 60


def assign_alert(row):
    if row['RUL_true'] <= CRITICAL_RUL:
        return CRITICAL
    elif row['RUL_true'] <= WARNING_RUL:
        return WARNING
    return NORMAL

def true_alert(rul):
    if rul <= CRITICAL_RUL:   
        return CRITICAL
    elif rul <= WARNING_RUL: 
        return WARNING
    return NORMAL
