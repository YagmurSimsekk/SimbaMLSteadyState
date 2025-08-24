from libsbml import SBMLReader
import logging


class SBMLParsingError(Exception):
    """Raised when SBML file cannot be parsed or contains errors."""
    pass


class UnsupportedSBMLVersionError(Exception):
    """Raised when SBML level/version combination is not supported."""
    pass

logger = logging.getLogger(__name__)

class MainSBMLParser:
    """
    Main SBML parser that detects SBML level/version and routes to appropriate parser.
    
    Supports commonly used SBML versions for ODE models:
    - Level 2: Version 4, 5
    - Level 3: Version 1, 2
    """
    
    # Define supported SBML level/version combinations
    SUPPORTED_VERSIONS = {
        (2, 4): "level_2.parser",
        (2, 5): "level_2.parser", 
        (3, 1): "level_3.parser",
        (3, 2): "level_3.parser"
    }
    
    def __init__(self, file_path):
        self.file_path = file_path
        self.level = None
        self.version = None
        self.model = None

    def detect_version_and_level(self):
        """
        Parse SBML file to detect level and version.
        
        Returns:
            tuple: (level, version, model) from the SBML document
            
        Raises:
            SBMLParsingError: If file cannot be parsed or contains errors
        """
        try:
            reader = SBMLReader()
            document = reader.readSBML(self.file_path)

            if document.getNumErrors() > 0:
                error_messages = []
                for i in range(document.getNumErrors()):
                    error_messages.append(document.getError(i).getMessage())
                logger.error(f"SBML parsing errors: {'; '.join(error_messages)}")
                raise SBMLParsingError(f"Error reading SBML file: {'; '.join(error_messages)}")

            model = document.getModel()
            if model is None:
                raise SBMLParsingError("No model found in SBML file.")

            level = document.getLevel()
            version = document.getVersion()
            
            self.level = level
            self.version = version
            self.model = model
            
            logger.info(f"Detected SBML Level {level}, Version {version}")
            return level, version, model
            
        except Exception as e:
            if isinstance(e, (SBMLParsingError, UnsupportedSBMLVersionError)):
                raise
            raise SBMLParsingError(f"Failed to parse SBML file '{self.file_path}': {str(e)}")

    def validate_ode_model(self, model):
        """
        Validate that the SBML model represents an ODE system.
        
        Args:
            model: SBML model object
            
        Raises:
            SBMLParsingError: If model doesn't appear to be ODE-based
        """
        if model.getListOfReactions().size() == 0:
            logger.warning("No reactions found - this may not be a dynamic ODE model")
            
        # Check for basic ODE model requirements
        has_kinetic_laws = False
        for reaction in model.getListOfReactions():
            if reaction.getKineticLaw() is not None:
                has_kinetic_laws = True
                break
                
        if not has_kinetic_laws and model.getListOfReactions().size() > 0:
            logger.warning("Reactions found but no kinetic laws - this may not be suitable for ODE simulation")

    def get_parser_module(self, level, version):
        """
        Get the appropriate parser module for the given level/version.
        
        Args:
            level: SBML level
            version: SBML version
            
        Returns:
            str: Module path for the parser
            
        Raises:
            UnsupportedSBMLVersionError: If level/version combination is not supported
        """
        if (level, version) not in self.SUPPORTED_VERSIONS:
            supported_versions = [f"Level {l} Version {v}" for l, v in self.SUPPORTED_VERSIONS.keys()]
            raise UnsupportedSBMLVersionError(
                f"SBML Level {level} Version {version} is not supported. "
                f"Supported versions: {', '.join(supported_versions)}"
            )
        
        return self.SUPPORTED_VERSIONS[(level, version)]

    def process(self):
        """
        Main processing method that detects version and delegates to appropriate parser.
        
        Returns:
            Parsed model data structure
            
        Raises:
            UnsupportedSBMLVersionError: If SBML version is not supported
            SBMLParsingError: If parsing fails
        """
        level, version, model = self.detect_version_and_level()
        
        # Validate ODE model characteristics
        self.validate_ode_model(model)
        
        # Get and instantiate the appropriate parser
        parser_module_path = self.get_parser_module(level, version)
        
        try:
            if parser_module_path == "level_2.parser":
                from .level_2.parser import Parser as VersionParser
            elif parser_module_path == "level_3.parser":
                from .level_3.parser import Parser as VersionParser
            else:
                raise ImportError(f"Unknown parser module: {parser_module_path}")
                
            parser = VersionParser(self.file_path, level, version)
            parsed_data = parser.parse()
            
            # Add metadata with file path for units parsing
            if 'metadata' not in parsed_data:
                parsed_data['metadata'] = {}
            parsed_data['metadata']['sbml_file_path'] = self.file_path
            
            return parsed_data
            
        except ImportError as e:
            raise SBMLParsingError(f"Failed to import parser for Level {level} Version {version}: {str(e)}")
