"""Reconcile mapped-clock uncertainty without borrowing future arrivals."""
import math

def source_fresh(context, field, age_s=.25):
    stamp=context.get(field);at=context.get('at_ns')
    if type(stamp) is not int or type(at) is not int or min(stamp,at)<=0:return False
    age=(at-stamp)/1e9
    if 0<=age<=age_s:return True
    # A source mapped slightly ahead can still have been received before the
    # scan query. This is clock uncertainty, not permission to use later data.
    if age>=0:return False
    bound=context.get('clock_bound_s')
    if type(bound) not in (float,int) or not math.isfinite(bound) or not 0<=bound<=.04:return False
    receipt_field={'speed_source_ns':'speed_receipt_ns','motion_source_ns':'motion_receipt_ns',
                   'steering_source_ns':'steering_receipt_ns'}.get(field)
    receipt=context.get(receipt_field) if receipt_field else None
    return (type(receipt) is int and 0<receipt<=at and at-receipt<=age_s*1e9 and
            -age<=bound+.01)
