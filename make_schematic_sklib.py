from collections import defaultdict
from skidl import Pin, Part, Alias, SchLib, SKIDL, TEMPLATE

from skidl.pin import pin_types

SKIDL_lib_version = '0.0.1'

make_schematic = SchLib(tool=SKIDL).add_parts(*[
        Part(**{ 'name':'Battery_2xAAA', 'dest':TEMPLATE, 'tool':SKIDL, 'aliases':Alias({'Battery_2xAAA'}), 'ref_prefix':'BT', 'fplist':None, 'footprint':'Battery:BatteryHolder_Keystone_2468_2xAAA', 'keywords':None, 'description':'', 'datasheet':None, 'pins':[
            Pin(num='1',name='+',func=pin_types.UNSPEC),
            Pin(num='2',name='-',func=pin_types.UNSPEC)] }),
        Part(**{ 'name':'Jumper_3', 'dest':TEMPLATE, 'tool':SKIDL, 'aliases':Alias({'Jumper_3'}), 'ref_prefix':'JP', 'fplist':None, 'footprint':'Connector_PinHeader_2.54mm:PinHeader_1x03_P2.54mm_Vertical', 'keywords':None, 'description':'', 'datasheet':None, 'pins':[
            Pin(num='1',name='1',func=pin_types.UNSPEC),
            Pin(num='2',name='2',func=pin_types.UNSPEC),
            Pin(num='3',name='3',func=pin_types.UNSPEC)] }),
        Part(**{ 'name':'C', 'dest':TEMPLATE, 'tool':SKIDL, 'aliases':Alias({'C'}), 'ref_prefix':'C', 'fplist':None, 'footprint':'Capacitor_SMD:C_0805_2012Metric', 'keywords':None, 'description':'', 'datasheet':None, 'pins':[
            Pin(num='1',name='1',func=pin_types.UNSPEC),
            Pin(num='2',name='2',func=pin_types.UNSPEC)] }),
        Part(**{ 'name':'R', 'dest':TEMPLATE, 'tool':SKIDL, 'aliases':Alias({'R'}), 'ref_prefix':'R', 'fplist':None, 'footprint':'Resistor_SMD:R_0805_2012Metric', 'keywords':None, 'description':'', 'datasheet':None, 'pins':[
            Pin(num='1',name='1',func=pin_types.UNSPEC),
            Pin(num='2',name='2',func=pin_types.UNSPEC)] }),
        Part(**{ 'name':'Conn_Electrodes', 'dest':TEMPLATE, 'tool':SKIDL, 'aliases':Alias({'Conn_Electrodes'}), 'ref_prefix':'J', 'fplist':None, 'footprint':'Connector_PinHeader_2.54mm:PinHeader_1x03_P2.54mm_Vertical', 'keywords':None, 'description':'', 'datasheet':None, 'pins':[
            Pin(num='1',name='E_NEG',func=pin_types.UNSPEC),
            Pin(num='2',name='E_POS',func=pin_types.UNSPEC),
            Pin(num='3',name='E_REF',func=pin_types.UNSPEC)] }),
        Part(**{ 'name':'Conn_HIL', 'dest':TEMPLATE, 'tool':SKIDL, 'aliases':Alias({'Conn_HIL'}), 'ref_prefix':'J', 'fplist':None, 'footprint':'Connector_PinHeader_2.54mm:PinHeader_1x02_P2.54mm_Vertical', 'keywords':None, 'description':'', 'datasheet':None, 'pins':[
            Pin(num='1',name='SIG',func=pin_types.UNSPEC),
            Pin(num='2',name='GND',func=pin_types.UNSPEC)] }),
        Part(**{ 'name':'Jumper_2', 'dest':TEMPLATE, 'tool':SKIDL, 'aliases':Alias({'Jumper_2'}), 'ref_prefix':'JP', 'fplist':None, 'footprint':'Connector_PinHeader_2.54mm:PinHeader_1x02_P2.54mm_Vertical', 'keywords':None, 'description':'', 'datasheet':None, 'pins':[
            Pin(num='1',name='1',func=pin_types.UNSPEC),
            Pin(num='2',name='2',func=pin_types.UNSPEC)] }),
        Part(**{ 'name':'INA333', 'dest':TEMPLATE, 'tool':SKIDL, 'aliases':Alias({'INA333'}), 'ref_prefix':'U', 'fplist':None, 'footprint':'Package_SO:VSSOP-8_3x3mm_P0.65mm', 'keywords':None, 'description':'', 'datasheet':None, 'pins':[
            Pin(num='1',name='RG',func=pin_types.UNSPEC),
            Pin(num='2',name='VIN-',func=pin_types.UNSPEC),
            Pin(num='3',name='VIN+',func=pin_types.UNSPEC),
            Pin(num='4',name='V-',func=pin_types.UNSPEC),
            Pin(num='5',name='REF',func=pin_types.UNSPEC),
            Pin(num='6',name='VOUT',func=pin_types.UNSPEC),
            Pin(num='7',name='V+',func=pin_types.UNSPEC),
            Pin(num='8',name='RG_',func=pin_types.UNSPEC)] }),
        Part(**{ 'name':'TLV8542', 'dest':TEMPLATE, 'tool':SKIDL, 'aliases':Alias({'TLV8542'}), 'ref_prefix':'U', 'fplist':None, 'footprint':'Package_SO:SOIC-8_3.9x4.9mm_P1.27mm', 'keywords':None, 'description':'', 'datasheet':None, 'pins':[
            Pin(num='1',name='OUTA',func=pin_types.UNSPEC),
            Pin(num='2',name='-INA',func=pin_types.UNSPEC),
            Pin(num='3',name='+INA',func=pin_types.UNSPEC),
            Pin(num='4',name='V-',func=pin_types.UNSPEC),
            Pin(num='5',name='+INB',func=pin_types.UNSPEC),
            Pin(num='6',name='-INB',func=pin_types.UNSPEC),
            Pin(num='7',name='OUTB',func=pin_types.UNSPEC),
            Pin(num='8',name='V+',func=pin_types.UNSPEC)] }),
        Part(**{ 'name':'SN74LVC1G66', 'dest':TEMPLATE, 'tool':SKIDL, 'aliases':Alias({'SN74LVC1G66'}), 'ref_prefix':'U', 'fplist':None, 'footprint':'Package_TO_SOT_SMD:SOT-353_SC-70-5', 'keywords':None, 'description':'', 'datasheet':None, 'pins':[
            Pin(num='1',name='A',func=pin_types.UNSPEC),
            Pin(num='2',name='B',func=pin_types.UNSPEC),
            Pin(num='3',name='GND',func=pin_types.UNSPEC),
            Pin(num='4',name='C',func=pin_types.UNSPEC),
            Pin(num='5',name='VCC',func=pin_types.UNSPEC)] }),
        Part(**{ 'name':'TLV3691', 'dest':TEMPLATE, 'tool':SKIDL, 'aliases':Alias({'TLV3691'}), 'ref_prefix':'U', 'fplist':None, 'footprint':'Package_TO_SOT_SMD:SOT-353_SC-70-5', 'keywords':None, 'description':'', 'datasheet':None, 'pins':[
            Pin(num='1',name='IN+',func=pin_types.UNSPEC),
            Pin(num='2',name='GND',func=pin_types.UNSPEC),
            Pin(num='3',name='IN-',func=pin_types.UNSPEC),
            Pin(num='4',name='OUT',func=pin_types.UNSPEC),
            Pin(num='5',name='VCC',func=pin_types.UNSPEC)] }),
        Part(**{ 'name':'Nucleo32_CN3', 'dest':TEMPLATE, 'tool':SKIDL, 'aliases':Alias({'Nucleo32_CN3'}), 'ref_prefix':'J', 'fplist':None, 'footprint':'Connector_PinSocket_2.54mm:PinSocket_1x15_P2.54mm_Vertical', 'keywords':None, 'description':'', 'datasheet':None, 'pins':[
            Pin(num='1',name='D1',func=pin_types.UNSPEC),
            Pin(num='2',name='D0',func=pin_types.UNSPEC),
            Pin(num='3',name='RESET',func=pin_types.UNSPEC),
            Pin(num='4',name='GND',func=pin_types.UNSPEC),
            Pin(num='5',name='D2',func=pin_types.UNSPEC),
            Pin(num='6',name='D3',func=pin_types.UNSPEC),
            Pin(num='7',name='D4',func=pin_types.UNSPEC),
            Pin(num='8',name='D5',func=pin_types.UNSPEC),
            Pin(num='9',name='D6',func=pin_types.UNSPEC),
            Pin(num='10',name='D7',func=pin_types.UNSPEC),
            Pin(num='11',name='D8',func=pin_types.UNSPEC),
            Pin(num='12',name='D9',func=pin_types.UNSPEC),
            Pin(num='13',name='D10',func=pin_types.UNSPEC),
            Pin(num='14',name='D11',func=pin_types.UNSPEC),
            Pin(num='15',name='D12',func=pin_types.UNSPEC)] }),
        Part(**{ 'name':'Nucleo32_CN4', 'dest':TEMPLATE, 'tool':SKIDL, 'aliases':Alias({'Nucleo32_CN4'}), 'ref_prefix':'J', 'fplist':None, 'footprint':'Connector_PinSocket_2.54mm:PinSocket_1x15_P2.54mm_Vertical', 'keywords':None, 'description':'', 'datasheet':None, 'pins':[
            Pin(num='1',name='VIN',func=pin_types.UNSPEC),
            Pin(num='2',name='GND',func=pin_types.UNSPEC),
            Pin(num='3',name='RESET',func=pin_types.UNSPEC),
            Pin(num='4',name='+5V',func=pin_types.UNSPEC),
            Pin(num='5',name='A7',func=pin_types.UNSPEC),
            Pin(num='6',name='A6',func=pin_types.UNSPEC),
            Pin(num='7',name='A5',func=pin_types.UNSPEC),
            Pin(num='8',name='A4',func=pin_types.UNSPEC),
            Pin(num='9',name='A3',func=pin_types.UNSPEC),
            Pin(num='10',name='A2',func=pin_types.UNSPEC),
            Pin(num='11',name='A1',func=pin_types.UNSPEC),
            Pin(num='12',name='A0',func=pin_types.UNSPEC),
            Pin(num='13',name='AREF',func=pin_types.UNSPEC),
            Pin(num='14',name='+3V3',func=pin_types.UNSPEC),
            Pin(num='15',name='D13',func=pin_types.UNSPEC)] }),
        Part(**{ 'name':'TestPoint', 'dest':TEMPLATE, 'tool':SKIDL, 'aliases':Alias({'TestPoint'}), 'ref_prefix':'TP', 'fplist':None, 'footprint':'TestPoint:TestPoint_Pad_D1.5mm', 'keywords':None, 'description':'', 'datasheet':None, 'pins':[
            Pin(num='1',name='1',func=pin_types.UNSPEC)] })])