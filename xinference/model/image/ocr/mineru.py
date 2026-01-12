# Copyright 2022-2025 XProbe Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import io
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import PIL.Image

if TYPE_CHECKING:
    from ..core import ImageModelFamilyV2

from .ocr_family import OCRModel

logger = logging.getLogger(__name__)


class MinerUModel(OCRModel):
    """MinerU 2.5 document parsing model for PDF to Markdown/JSON conversion.
    
    MinerU is a high-precision document parsing tool that converts complex PDF 
    documents into machine-readable formats (Markdown, JSON). It supports:
    - Layout detection and preservation
    - Table extraction (HTML format)
    - Formula extraction (LaTeX format)
    - Multi-language OCR (109 languages)
    - Image extraction with descriptions
    
    Note: MinerU manages its own model downloads through the mineru package.
    No external model download from HuggingFace/ModelScope is required.
    """

    required_libs = ("mineru",)
    
    # MinerU manages its own models, no need for xinference to download
    skip_model_download = True

    @classmethod
    def match(cls, model_family: "ImageModelFamilyV2") -> bool:
        return model_family.model_name == "MinerU"

    def __init__(
        self,
        model_uid: str,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        model_spec: Optional["ImageModelFamilyV2"] = None,
        **kwargs,
    ):
        self.model_family = model_spec
        self._model_uid = model_uid
        # MinerU doesn't use model_path - it manages models internally
        self._model_path = model_path
        self._device = device
        # Model info
        self._model_spec = model_spec
        self._abilities = model_spec.model_ability if model_spec else ["ocr", "document-parsing"]
        self._kwargs = kwargs
        # MinerU specific settings from default_model_config or kwargs
        default_config = {}
        if model_spec and hasattr(model_spec, 'default_model_config') and model_spec.default_model_config:
            default_config = model_spec.default_model_config
        self._backend = kwargs.get("backend", default_config.get("backend", "hybrid-auto-engine"))
        self._parse_method = kwargs.get("parse_method", default_config.get("parse_method", "auto"))
        self._language = kwargs.get("language", default_config.get("language", "ch"))
        self._loaded = False

    @property
    def model_ability(self):
        return self._abilities

    def load(self):
        """Initialize MinerU environment and verify dependencies."""
        try:
            # Import MinerU modules to verify installation
            from mineru.cli.common import prepare_env, read_fn
            from mineru.data.data_reader_writer import FileBasedDataWriter
            from mineru.utils.enum_class import MakeMode
            
            logger.info("MinerU dependencies verified successfully")
            self._loaded = True
            
        except ImportError as e:
            logger.error(f"Failed to import MinerU: {e}")
            raise RuntimeError(
                "MinerU is not installed. Please install it with: "
                "pip install 'mineru[all]>=2.5.0'"
            ) from e

    def _convert_image_to_pdf_bytes(self, image: PIL.Image.Image) -> bytes:
        """Convert a PIL Image to PDF bytes for processing."""
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.pdfgen import canvas
            
            # Save image to bytes
            img_buffer = io.BytesIO()
            if image.mode in ["RGBA", "CMYK"]:
                image = image.convert("RGB")
            image.save(img_buffer, format="PNG")
            img_buffer.seek(0)
            
            # Create PDF with the image
            pdf_buffer = io.BytesIO()
            c = canvas.Canvas(pdf_buffer, pagesize=letter)
            
            # Calculate dimensions to fit the page
            page_width, page_height = letter
            img_width, img_height = image.size
            
            # Scale to fit page while maintaining aspect ratio
            scale = min(page_width / img_width, page_height / img_height) * 0.9
            new_width = img_width * scale
            new_height = img_height * scale
            
            # Center the image
            x = (page_width - new_width) / 2
            y = (page_height - new_height) / 2
            
            # Save temp image and draw it
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                image.save(tmp, format="PNG")
                tmp_path = tmp.name
            
            try:
                c.drawImage(tmp_path, x, y, new_width, new_height)
                c.save()
            finally:
                os.unlink(tmp_path)
            
            pdf_buffer.seek(0)
            return pdf_buffer.read()
            
        except ImportError:
            # Fallback: use PIL's built-in PDF saving (lower quality)
            pdf_buffer = io.BytesIO()
            if image.mode in ["RGBA", "CMYK"]:
                image = image.convert("RGB")
            image.save(pdf_buffer, format="PDF")
            pdf_buffer.seek(0)
            return pdf_buffer.read()

    def _parse_pdf(
        self,
        pdf_bytes: bytes,
        output_dir: str,
        file_name: str = "document",
        **kwargs,
    ) -> Dict[str, Any]:
        """Parse PDF using MinerU backend.
        
        Args:
            pdf_bytes: PDF file content as bytes
            output_dir: Directory to store output files
            file_name: Base name for output files
            **kwargs: Additional parsing options
            
        Returns:
            Dictionary containing parsing results
        """
        from mineru.cli.common import convert_pdf_bytes_to_bytes_by_pypdfium2, prepare_env
        from mineru.data.data_reader_writer import FileBasedDataWriter
        from mineru.utils.enum_class import MakeMode
        from mineru.utils.engine_utils import get_vlm_engine
        
        backend = kwargs.get("backend", self._backend)
        parse_method = kwargs.get("parse_method", self._parse_method)
        language = kwargs.get("language", self._language)
        output_format = kwargs.get("output_format", "markdown")
        start_page_id = kwargs.get("start_page_id", 0)
        end_page_id = kwargs.get("end_page_id", None)
        formula_enable = kwargs.get("formula_enable", True)
        table_enable = kwargs.get("table_enable", True)
        
        # Convert pages if needed
        pdf_bytes = convert_pdf_bytes_to_bytes_by_pypdfium2(
            pdf_bytes, start_page_id, end_page_id
        )
        
        # Prepare output directories
        local_image_dir, local_md_dir = prepare_env(output_dir, file_name, parse_method)
        image_writer = FileBasedDataWriter(local_image_dir)
        md_writer = FileBasedDataWriter(local_md_dir)
        
        result = {
            "markdown": "",
            "json": None,
            "content_list": None,
            "output_dir": local_md_dir,
            "image_dir": local_image_dir,
        }
        
        if backend == "pipeline":
            from mineru.backend.pipeline.pipeline_analyze import doc_analyze as pipeline_doc_analyze
            from mineru.backend.pipeline.model_json_to_middle_json import result_to_middle_json as pipeline_result_to_middle_json
            from mineru.backend.pipeline.pipeline_middle_json_mkcontent import union_make as pipeline_union_make
            
            infer_results, all_image_lists, all_pdf_docs, lang_list, ocr_enabled_list = pipeline_doc_analyze(
                [pdf_bytes], 
                [language], 
                parse_method=parse_method,
                formula_enable=formula_enable,
                table_enable=table_enable
            )
            
            if infer_results:
                model_list = infer_results[0]
                images_list = all_image_lists[0]
                pdf_doc = all_pdf_docs[0]
                _lang = lang_list[0]
                _ocr_enable = ocr_enabled_list[0]
                
                middle_json = pipeline_result_to_middle_json(
                    model_list, images_list, pdf_doc, image_writer, 
                    _lang, _ocr_enable, formula_enable
                )
                
                pdf_info = middle_json["pdf_info"]
                image_dir_name = os.path.basename(local_image_dir)
                
                # Generate markdown
                md_content = pipeline_union_make(pdf_info, MakeMode.MM_MD, image_dir_name)
                result["markdown"] = md_content
                result["json"] = middle_json
                
                # Generate content list
                content_list = pipeline_union_make(pdf_info, MakeMode.CONTENT_LIST, image_dir_name)
                result["content_list"] = content_list
                
        elif backend.startswith("vlm-"):
            from mineru.backend.vlm.vlm_analyze import doc_analyze as vlm_doc_analyze
            from mineru.backend.vlm.vlm_middle_json_mkcontent import union_make as vlm_union_make
            
            actual_backend = backend[4:]  # Remove "vlm-" prefix
            if actual_backend == "auto-engine":
                actual_backend = get_vlm_engine(inference_engine='auto', is_async=False)
            
            middle_json, infer_result = vlm_doc_analyze(
                pdf_bytes, 
                image_writer=image_writer, 
                backend=actual_backend
            )
            
            pdf_info = middle_json["pdf_info"]
            image_dir_name = os.path.basename(local_image_dir)
            
            md_content = vlm_union_make(pdf_info, MakeMode.MM_MD, image_dir_name)
            result["markdown"] = md_content
            result["json"] = middle_json
            
            content_list = vlm_union_make(pdf_info, MakeMode.CONTENT_LIST, image_dir_name)
            result["content_list"] = content_list
            
        elif backend.startswith("hybrid-"):
            from mineru.backend.hybrid.hybrid_analyze import doc_analyze as hybrid_doc_analyze
            from mineru.backend.vlm.vlm_middle_json_mkcontent import union_make as vlm_union_make
            
            actual_backend = backend[7:]  # Remove "hybrid-" prefix
            if actual_backend == "auto-engine":
                actual_backend = get_vlm_engine(inference_engine='auto', is_async=False)
            
            actual_parse_method = f"hybrid_{parse_method}"
            
            middle_json, infer_result, _vlm_ocr_enable = hybrid_doc_analyze(
                pdf_bytes,
                image_writer=image_writer,
                backend=actual_backend,
                parse_method=actual_parse_method,
                language=language,
                inline_formula_enable=formula_enable,
            )
            
            pdf_info = middle_json["pdf_info"]
            image_dir_name = os.path.basename(local_image_dir)
            
            md_content = vlm_union_make(pdf_info, MakeMode.MM_MD, image_dir_name)
            result["markdown"] = md_content
            result["json"] = middle_json
            
            content_list = vlm_union_make(pdf_info, MakeMode.CONTENT_LIST, image_dir_name)
            result["content_list"] = content_list
        else:
            raise ValueError(f"Unknown backend: {backend}")
        
        # Save outputs
        if result["markdown"]:
            md_writer.write_string(f"{file_name}.md", result["markdown"])
        if result["json"]:
            md_writer.write_string(
                f"{file_name}_middle.json", 
                json.dumps(result["json"], ensure_ascii=False, indent=4)
            )
        if result["content_list"]:
            md_writer.write_string(
                f"{file_name}_content_list.json",
                json.dumps(result["content_list"], ensure_ascii=False, indent=4)
            )
        
        return result

    def ocr(
        self,
        image_or_pdf: Union[PIL.Image.Image, bytes, str, Path],
        **kwargs,
    ) -> Union[str, Dict[str, Any]]:
        """Perform document parsing using MinerU.
        
        Args:
            image_or_pdf: Input can be:
                - PIL.Image.Image: Will be converted to PDF for processing
                - bytes: Raw PDF bytes
                - str/Path: Path to PDF or image file
            **kwargs: Additional parameters:
                - backend: "pipeline", "vlm-auto-engine", "hybrid-auto-engine" (default)
                - parse_method: "auto" (default), "txt", "ocr"
                - language: "ch" (default), "en", "korean", "japan", etc.
                - output_format: "markdown" (default), "json"
                - return_dict: Whether to return full result dict (default: False)
                - output_dir: Custom output directory (optional)
                - start_page_id: Start page (0-indexed, default: 0)
                - end_page_id: End page (optional, default: all pages)
                - formula_enable: Enable formula extraction (default: True)
                - table_enable: Enable table extraction (default: True)
        
        Returns:
            Markdown string by default, or dict with full results if return_dict=True
        """
        if not self._loaded:
            self.load()
        
        logger.info(f"MinerU OCR called with kwargs: {kwargs}")
        
        return_dict = kwargs.get("return_dict", False)
        output_format = kwargs.get("output_format", "markdown")
        output_dir = kwargs.get("output_dir", None)
        
        # Determine input type and convert to PDF bytes
        pdf_bytes = None
        file_name = "document"
        
        if isinstance(image_or_pdf, PIL.Image.Image):
            logger.info("Converting PIL Image to PDF for processing")
            pdf_bytes = self._convert_image_to_pdf_bytes(image_or_pdf)
            file_name = "image_document"
            
        elif isinstance(image_or_pdf, bytes):
            pdf_bytes = image_or_pdf
            file_name = "uploaded_document"
            
        elif isinstance(image_or_pdf, (str, Path)):
            file_path = Path(image_or_pdf)
            file_name = file_path.stem
            
            if file_path.suffix.lower() in [".pdf"]:
                with open(file_path, "rb") as f:
                    pdf_bytes = f.read()
            elif file_path.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"]:
                image = PIL.Image.open(file_path)
                pdf_bytes = self._convert_image_to_pdf_bytes(image)
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
        else:
            raise ValueError(
                f"Unsupported input type: {type(image_or_pdf)}. "
                "Expected PIL.Image, bytes, str, or Path."
            )
        
        # Create temporary output directory if not specified
        if output_dir is None:
            output_dir = tempfile.mkdtemp(prefix="mineru_output_")
        
        try:
            result = self._parse_pdf(
                pdf_bytes=pdf_bytes,
                output_dir=output_dir,
                file_name=file_name,
                **kwargs,
            )
            
            if return_dict:
                return {
                    "text": result["markdown"],
                    "markdown": result["markdown"],
                    "json": result["json"],
                    "content_list": result["content_list"],
                    "output_dir": result["output_dir"],
                    "image_dir": result["image_dir"],
                    "model": "MinerU",
                    "success": True,
                }
            
            if output_format == "json":
                return json.dumps(result["json"], ensure_ascii=False, indent=2)
            else:
                return result["markdown"]
                
        except Exception as e:
            logger.error(f"MinerU parsing error: {e}")
            if return_dict:
                return {
                    "text": "",
                    "error": str(e),
                    "model": "MinerU",
                    "success": False,
                }
            raise
