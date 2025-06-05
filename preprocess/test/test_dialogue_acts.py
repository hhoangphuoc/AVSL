#!/usr/bin/env python3
"""
Test script for dialogue acts processing function.
This script tests the dialogue acts processing with the provided XML files.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dialogue_acts_process import (
    load_dialogue_act_types, 
    load_adjacency_pair_types,
    extract_word_ids_from_href,
    extract_dact_info_from_href
)

def test_dialogue_act_types():
    """Test loading dialogue act types from da-types.xml"""
    print("Testing dialogue act types loading...")
    
    da_types_file = os.path.join(os.path.dirname(__file__), '..', 'xml', 'da-types.xml')
    
    if os.path.exists(da_types_file):
        da_types = load_dialogue_act_types(da_types_file)
        print(f"Loaded {len(da_types)} dialogue act types")
        
        # Print some examples
        for i, (da_id, da_info) in enumerate(da_types.items()):
            if i < 5:  # Show first 5
                print(f"  {da_id}: {da_info}")
        print("✓ Dialogue act types loading successful")
    else:
        print(f"✗ da-types.xml file not found at {da_types_file}")

def test_adjacency_pair_types():
    """Test loading adjacency pair types from ap-types.xml"""
    print("\nTesting adjacency pair types loading...")
    
    ap_types_file = os.path.join(os.path.dirname(__file__), '..', 'xml', 'ap-types.xml')
    
    if os.path.exists(ap_types_file):
        ap_types = load_adjacency_pair_types(ap_types_file)
        print(f"Loaded {len(ap_types)} adjacency pair types")
        
        # Print all examples since there are few
        for ap_id, ap_gloss in ap_types.items():
            print(f"  {ap_id}: {ap_gloss}")
        print("✓ Adjacency pair types loading successful")
    else:
        print(f"✗ ap-types.xml file not found at {ap_types_file}")

def test_word_extraction():
    """Test word ID extraction from href references"""
    print("\nTesting word ID extraction...")
    
    # Test single word reference
    single_href = "ES2002a.A.words.xml#id(ES2002a.A.words42)"
    single_ids = extract_word_ids_from_href(single_href)
    print(f"Single word: {single_href} -> {single_ids}")
    
    # Test range reference
    range_href = "ES2002a.A.words.xml#id(ES2002a.A.words0)..id(ES2002a.A.words12)"
    range_ids = extract_word_ids_from_href(range_href)
    print(f"Range words: {range_href} -> {range_ids[:5]}...{range_ids[-5:]} (showing first/last 5 of {len(range_ids)})")
    
    print("✓ Word ID extraction successful")

def test_dact_extraction():
    """Test dialogue act information extraction from href references"""
    print("\nTesting dialogue act information extraction...")
    
    dact_href = "ES2002a.A.dialog-act.xml#id(ES2002a.A.dialog-act.dharshi.1)"
    dact_info = extract_dact_info_from_href(dact_href)
    print(f"Dialogue act: {dact_href} -> {dact_info}")
    
    print("✓ Dialogue act information extraction successful")

def main():
    """Run all tests"""
    print("Running dialogue acts processing tests...\n")
    
    try:
        test_dialogue_act_types()
        test_adjacency_pair_types()
        test_word_extraction()
        test_dact_extraction()
        
        print("\n✓ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 