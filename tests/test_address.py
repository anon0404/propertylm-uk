from propertylm.utils.address import address_keys

def test_hierarchy_keys():
    k = address_keys("16 Alesmore WS13 8FD")
    assert k.postcode == "WS13 8FD"
    assert k.sector == "WS13 8"
    assert k.district == "WS13"
    assert k.number == "16"
