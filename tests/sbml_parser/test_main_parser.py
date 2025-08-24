"""
Tests for the main SBML parser functionality.
"""
import pytest
import tempfile
import os
from simba_ml.sbml_parser.main_parser import MainSBMLParser, SBMLParsingError, UnsupportedSBMLVersionError


class TestMainSBMLParser:
    """Test the main SBML parser functionality."""
    
    def test_init(self):
        """Test parser initialization."""
        parser = MainSBMLParser("test_file.xml")
        assert parser.file_path == "test_file.xml"
        assert parser.level is None
        assert parser.version is None
        assert parser.model is None

    def test_supported_versions(self):
        """Test that supported versions are correctly defined."""
        expected_versions = {
            (2, 4): "level_2.parser",
            (2, 5): "level_2.parser", 
            (3, 1): "level_3.parser",
            (3, 2): "level_3.parser"
        }
        assert MainSBMLParser.SUPPORTED_VERSIONS == expected_versions

    def test_get_parser_module_supported(self):
        """Test getting parser module for supported versions."""
        parser = MainSBMLParser("test.xml")
        
        # Test Level 2 versions
        assert parser.get_parser_module(2, 4) == "level_2.parser"
        assert parser.get_parser_module(2, 5) == "level_2.parser"
        
        # Test Level 3 versions  
        assert parser.get_parser_module(3, 1) == "level_3.parser"
        assert parser.get_parser_module(3, 2) == "level_3.parser"

    def test_get_parser_module_unsupported(self):
        """Test error for unsupported versions."""
        parser = MainSBMLParser("test.xml")
        
        with pytest.raises(UnsupportedSBMLVersionError) as excinfo:
            parser.get_parser_module(1, 2)
        
        assert "Level 1 Version 2 is not supported" in str(excinfo.value)
        assert "Supported versions:" in str(excinfo.value)

    def test_detect_version_and_level_invalid_file(self):
        """Test error handling for invalid file."""
        parser = MainSBMLParser("nonexistent_file.xml")
        
        with pytest.raises(SBMLParsingError):
            parser.detect_version_and_level()

    def test_validate_ode_model_no_reactions(self):
        """Test validation warning for models without reactions."""
        # Create a minimal SBML model without reactions for testing
        sbml_content = '''<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
    <model id="test_model">
        <listOfCompartments>
            <compartment id="cell" constant="true" spatialDimensions="3" size="1"/>
        </listOfCompartments>
        <listOfSpecies>
            <species id="A" compartment="cell" hasOnlySubstanceUnits="false" 
                     boundaryCondition="false" constant="false"/>
        </listOfSpecies>
    </model>
</sbml>'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
            f.write(sbml_content)
            f.flush()
            
            try:
                parser = MainSBMLParser(f.name)
                level, version, model = parser.detect_version_and_level()
                
                # Should not raise error, just log warning
                parser.validate_ode_model(model)
                
                assert level == 3
                assert version == 1
                assert model is not None
                
            finally:
                os.unlink(f.name)

    def test_detect_version_level_with_valid_sbml(self):
        """Test detection with valid SBML content."""
        sbml_content = '''<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
    <model id="test_model">
        <listOfCompartments>
            <compartment id="cell" constant="true" spatialDimensions="3" size="1"/>
        </listOfCompartments>
        <listOfSpecies>
            <species id="A" compartment="cell" hasOnlySubstanceUnits="false" 
                     boundaryCondition="false" constant="false" initialConcentration="1.0"/>
            <species id="B" compartment="cell" hasOnlySubstanceUnits="false" 
                     boundaryCondition="false" constant="false" initialConcentration="0.0"/>
        </listOfSpecies>
        <listOfReactions>
            <reaction id="R1" reversible="false">
                <listOfReactants>
                    <speciesReference species="A" stoichiometry="1" constant="true"/>
                </listOfReactants>
                <listOfProducts>
                    <speciesReference species="B" stoichiometry="1" constant="true"/>
                </listOfProducts>
                <kineticLaw>
                    <math xmlns="http://www.w3.org/1998/Math/MathML">
                        <apply>
                            <times/>
                            <ci>k</ci>
                            <ci>A</ci>
                        </apply>
                    </math>
                    <listOfLocalParameters>
                        <localParameter id="k" value="0.1"/>
                    </listOfLocalParameters>
                </kineticLaw>
            </reaction>
        </listOfReactions>
    </model>
</sbml>'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
            f.write(sbml_content)
            f.flush()
            
            try:
                parser = MainSBMLParser(f.name)
                level, version, model = parser.detect_version_and_level()
                
                assert level == 3
                assert version == 2
                assert model is not None
                assert parser.level == 3
                assert parser.version == 2
                assert parser.model is not None
                
            finally:
                os.unlink(f.name)


class TestSBMLParsingIntegration:
    """Integration tests for SBML parsing."""
    
    def create_test_sbml_file(self, level, version, content_additions=""):
        """Helper to create test SBML files."""
        base_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level{level}/version{version}/core" level="{level}" version="{version}">
    <model id="test_model" name="Test Model">
        <listOfCompartments>
            <compartment id="cell" constant="true" spatialDimensions="3" size="1.0"/>
        </listOfCompartments>
        <listOfSpecies>
            <species id="A" compartment="cell" hasOnlySubstanceUnits="false" 
                     boundaryCondition="false" constant="false" initialConcentration="1.0"/>
        </listOfSpecies>
        <listOfReactions>
            <reaction id="R1" reversible="false">
                <listOfReactants>
                    <speciesReference species="A" stoichiometry="1" constant="true"/>
                </listOfReactants>
                <kineticLaw>
                    <math xmlns="http://www.w3.org/1998/Math/MathML">
                        <apply>
                            <times/>
                            <ci>k</ci>
                            <ci>A</ci>
                        </apply>
                    </math>
                    {'<listOfLocalParameters><localParameter id="k" value="0.1"/></listOfLocalParameters>' if level == 3 else '<listOfParameters><parameter id="k" value="0.1"/></listOfParameters>'}
                </kineticLaw>
            </reaction>
        </listOfReactions>
        {content_additions}
    </model>
</sbml>'''
        
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False)
        temp_file.write(base_content)
        temp_file.flush()
        temp_file.close()
        return temp_file.name

    def test_level_2_version_4_parsing(self):
        """Test parsing Level 2 Version 4 files."""
        test_file = self.create_test_sbml_file(2, 4)
        
        try:
            parser = MainSBMLParser(test_file)
            result = parser.process()
            
            assert isinstance(result, dict)
            assert 'sbml_info' in result
            assert result['sbml_info']['level'] == 2
            assert result['sbml_info']['version'] == 4
            
        finally:
            os.unlink(test_file)

    def test_level_2_version_5_parsing(self):
        """Test parsing Level 2 Version 5 files."""
        test_file = self.create_test_sbml_file(2, 5)
        
        try:
            parser = MainSBMLParser(test_file)
            result = parser.process()
            
            assert isinstance(result, dict)
            assert 'sbml_info' in result
            assert result['sbml_info']['level'] == 2
            assert result['sbml_info']['version'] == 5
            
        finally:
            os.unlink(test_file)

    def test_level_3_version_1_parsing(self):
        """Test parsing Level 3 Version 1 files."""
        test_file = self.create_test_sbml_file(3, 1)
        
        try:
            parser = MainSBMLParser(test_file)
            result = parser.process()
            
            assert isinstance(result, dict)
            assert 'sbml_info' in result
            assert result['sbml_info']['level'] == 3
            assert result['sbml_info']['version'] == 1
            
        finally:
            os.unlink(test_file)

    def test_level_3_version_2_parsing(self):
        """Test parsing Level 3 Version 2 files."""
        test_file = self.create_test_sbml_file(3, 2)
        
        try:
            parser = MainSBMLParser(test_file)
            result = parser.process()
            
            assert isinstance(result, dict)
            assert 'sbml_info' in result
            assert result['sbml_info']['level'] == 3
            assert result['sbml_info']['version'] == 2
            
        finally:
            os.unlink(test_file)

    def test_unsupported_version_raises_error(self):
        """Test that unsupported versions raise appropriate errors."""
        # Create a Level 1 SBML file (unsupported)
        sbml_content = '''<?xml version="1.0" encoding="UTF-8"?>
<sbml level="1" version="2">
    <model>
        <listOfCompartments>
            <compartment name="cell" volume="1"/>
        </listOfCompartments>
    </model>
</sbml>'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
            f.write(sbml_content)
            f.flush()
            
            try:
                parser = MainSBMLParser(f.name)
                
                with pytest.raises(UnsupportedSBMLVersionError) as excinfo:
                    parser.process()
                
                assert "Level 1 Version 2 is not supported" in str(excinfo.value)
                
            finally:
                os.unlink(f.name)

    def test_malformed_sbml_raises_error(self):
        """Test that malformed SBML raises parsing error."""
        malformed_content = '''<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
    <model id="test_model">
        <listOfSpecies>
            <species id="A" compartment="nonexistent_compartment"/>
        </listOfSpecies>
    </model>
</sbml>'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
            f.write(malformed_content)
            f.flush()
            
            try:
                parser = MainSBMLParser(f.name)
                # Should not raise error during parsing, but might log warnings
                result = parser.process()
                assert isinstance(result, dict)
                
            finally:
                os.unlink(f.name)

    def teardown_method(self):
        """Clean up any temporary files."""
        pass


if __name__ == "__main__":
    pytest.main([__file__])