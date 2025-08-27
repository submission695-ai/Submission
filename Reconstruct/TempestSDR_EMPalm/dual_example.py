import numpy as np
from scipy import signal
from scipy.fft import fft, ifft, fftfreq
from scipy.ndimage import median_filter
import cv2
from typing import Tuple, Dict, Optional, List
from numba import jit, prange
import warnings
warnings.filterwarnings('ignore')

class OptimizedDualModeSeparation:
    """
    Optimized separation and reconstruction of palmprint and palmvein images 
    from interleaved dual-mode transmission signals without any mixing
    """
    
    def __init__(self, frame_width: int, frame_height: int, 
                 modality_order: str = 'print_first',
                 sync_method: str = 'fft_correlation',
                 buffer_size: int = 1024*1024):
        """
        Initialize optimized dual-mode separation module
        
        Args:
            frame_width: Width of each frame in pixels
            frame_height: Height of each frame in pixels
            modality_order: 'print_first' if M_k=0 is palmprint, 'vein_first' if opposite
            sync_method: Method for frame synchronization ('fft_correlation', 'autocorr', 'energy')
            buffer_size: Size of internal processing buffer
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.frame_size = frame_width * frame_height
        self.modality_order = modality_order
        self.sync_method = sync_method
        self.buffer_size = buffer_size
        
        # Pre-allocate buffers for performance
        self.fft_buffer = np.zeros(self.buffer_size, dtype=np.complex128)
        self.correlation_buffer = np.zeros(self.buffer_size, dtype=np.float64)
        
        # Cache for frame boundaries
        self.cached_boundaries = None
        self.cache_valid = False
        
    @staticmethod
    @jit(nopython=True, parallel=True)
    def _fast_magnitude(iq_data: np.ndarray) -> np.ndarray:
        """
        Fast magnitude computation using Numba JIT
        
        Args:
            iq_data: Complex IQ data
            
        Returns:
            Magnitude array
        """
        magnitude = np.zeros(len(iq_data), dtype=np.float64)
        for i in prange(len(iq_data)):
            magnitude[i] = np.abs(iq_data[i])
        return magnitude
    
    def detect_frame_boundaries_fft(self, iq_data: np.ndarray) -> List[int]:
        """
        Detect frame boundaries using FFT-based autocorrelation (inspired by fft.c)
        
        Args:
            iq_data: Continuous IQ data stream
            
        Returns:
            List of frame starting indices
        """
        # Use cached boundaries if available
        if self.cache_valid and self.cached_boundaries is not None:
            return self.cached_boundaries
        
        # Convert to magnitude for processing
        magnitude = self._fast_magnitude(iq_data)
        
        # Limit processing to reasonable size for efficiency
        process_size = min(len(magnitude), 10 * self.frame_size)
        mag_segment = magnitude[:process_size]
        
        # FFT-based autocorrelation (similar to fft_autocorrelation in fft.c)
        fft_size = 2 ** int(np.ceil(np.log2(len(mag_segment))))
        
        # Pad to power of 2 for FFT efficiency
        padded = np.pad(mag_segment, (0, fft_size - len(mag_segment)), mode='constant')
        
        # Compute autocorrelation via FFT
        fft_result = fft(padded)
        power_spectrum = np.abs(fft_result) ** 2
        autocorr = np.real(ifft(power_spectrum))
        
        # Normalize autocorrelation
        autocorr = autocorr[:len(mag_segment)]
        autocorr = autocorr / autocorr[0]
        
        # Find peaks corresponding to frame period
        min_distance = int(self.frame_size * 0.8)
        max_distance = int(self.frame_size * 1.2)
        
        # Search for strongest periodic component
        search_region = autocorr[min_distance:max_distance]
        if len(search_region) > 0:
            peak_idx = np.argmax(search_region) + min_distance
            estimated_period = peak_idx
        else:
            estimated_period = self.frame_size
        
        # Generate frame boundaries based on detected period
        boundaries = list(range(0, len(iq_data), estimated_period))
        
        # Cache the result
        self.cached_boundaries = boundaries
        self.cache_valid = True
        
        return boundaries
    
    def extract_frames_direct(self, iq_data: np.ndarray, 
                             frame_boundaries: Optional[List[int]] = None) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Directly extract and separate frames without intermediate mixing
        
        Args:
            iq_data: Continuous IQ data stream
            frame_boundaries: Optional pre-computed frame boundaries
            
        Returns:
            Tuple of (palmprint_frames, palmvein_frames) - already separated
        """
        if frame_boundaries is None:
            frame_boundaries = self.detect_frame_boundaries_fft(iq_data)
        
        palmprint_frames = []
        palmvein_frames = []
        
        # Direct separation during extraction - no mixing step
        for k in range(len(frame_boundaries) - 1):
            start_idx = frame_boundaries[k]
            end_idx = frame_boundaries[k + 1]
            
            frame_data = iq_data[start_idx:end_idx]
            
            # Pad or truncate to exact frame size
            if len(frame_data) < self.frame_size:
                frame_data = np.pad(frame_data, (0, self.frame_size - len(frame_data)), 
                                   mode='constant')
            elif len(frame_data) > self.frame_size:
                frame_data = frame_data[:self.frame_size]
            
            # Direct modality assignment based on frame parity
            M_k = k % 2
            
            if self.modality_order == 'print_first':
                if M_k == 0:
                    palmprint_frames.append(frame_data)
                else:
                    palmvein_frames.append(frame_data)
            else:  # vein_first
                if M_k == 0:
                    palmvein_frames.append(frame_data)
                else:
                    palmprint_frames.append(frame_data)
        
        return palmprint_frames, palmvein_frames
    
    @staticmethod
    @jit(nopython=True, parallel=True)
    def _fast_accumulate(frames: List[np.ndarray], height: int, width: int) -> np.ndarray:
        """
        Fast frame accumulation using Numba JIT
        
        Args:
            frames: List of frames to accumulate
            height: Image height
            width: Image width
            
        Returns:
            Accumulated image
        """
        accumulated = np.zeros((height, width), dtype=np.complex128)
        num_frames = len(frames)
        
        for frame in frames:
            frame_2d = frame.reshape(height, width)
            accumulated += frame_2d
        
        if num_frames > 0:
            accumulated /= num_frames
            
        return np.abs(accumulated)
    
    def reconstruct_modality_optimized(self, iq_frames: List[np.ndarray], 
                                      modality_type: str = 'palmprint') -> np.ndarray:
        """
        Optimized reconstruction for a single modality without mixing
        
        Args:
            iq_frames: List of IQ frames for a single modality (already separated)
            modality_type: 'palmprint' or 'palmvein' for specific optimization
            
        Returns:
            Reconstructed image for the specific modality
        """
        if len(iq_frames) == 0:
            return np.zeros((self.frame_height, self.frame_width))
        
        # Convert list to numpy array for vectorized operations
        frames_array = np.array(iq_frames)
        
        # Vectorized accumulation and averaging
        accumulated = np.mean(frames_array, axis=0)
        
        # Reshape to 2D image
        image_2d = accumulated.reshape(self.frame_height, self.frame_width)
        
        # Convert to magnitude
        reconstructed = np.abs(image_2d)
        
        # Normalize
        if np.max(reconstructed) > 0:
            reconstructed = reconstructed / np.max(reconstructed)
        
        # Apply modality-specific enhancement
        enhanced = self.apply_modality_enhancement(reconstructed, modality_type)
        
        return enhanced
    
    def apply_modality_enhancement(self, image: np.ndarray, 
                                  modality_type: str) -> np.ndarray:
        """
        Apply modality-specific enhancement without mixing
        
        Args:
            image: Input image for a single modality
            modality_type: 'palmprint' or 'palmvein'
            
        Returns:
            Enhanced image
        """
        if modality_type == 'palmprint':
            # Palmprint-specific enhancement
            enhanced = self._enhance_palmprint_features(image)
        elif modality_type == 'palmvein':
            # Palmvein-specific enhancement  
            enhanced = self._enhance_palmvein_features(image)
        else:
            enhanced = image.copy()
        
        return np.clip(enhanced, 0, 1)
    
    def _enhance_palmprint_features(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance palmprint ridge patterns without mixing
        
        Args:
            image: Palmprint image
            
        Returns:
            Enhanced palmprint
        """
        # Denoise
        denoised = median_filter(image, size=3)
        
        # Ridge enhancement using directional filters
        # Horizontal ridges
        kernel_h = np.array([[-1, -1, -1],
                            [ 2,  2,  2],
                            [-1, -1, -1]]) / 6
        
        # Vertical ridges
        kernel_v = np.array([[-1,  2, -1],
                            [-1,  2, -1],
                            [-1,  2, -1]]) / 6
        
        # Apply directional filters
        enhanced_h = cv2.filter2D(denoised, -1, kernel_h)
        enhanced_v = cv2.filter2D(denoised, -1, kernel_v)
        
        # Combine directional enhancements
        enhanced = np.maximum(enhanced_h, enhanced_v)
        
        # CLAHE for contrast
        enhanced_uint8 = (enhanced * 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced_uint8 = clahe.apply(enhanced_uint8)
        
        return enhanced_uint8.astype(np.float64) / 255.0
    
    def _enhance_palmvein_features(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance palmvein vascular patterns without mixing
        
        Args:
            image: Palmvein image
            
        Returns:
            Enhanced palmvein
        """
        # Denoise with edge preservation
        denoised = cv2.bilateralFilter((image * 255).astype(np.uint8), 5, 50, 50)
        denoised = denoised.astype(np.float64) / 255.0
        
        # Vessel enhancement using multi-scale approach
        scales = [3, 5, 7]
        vessel_responses = []
        
        for scale in scales:
            # Top-hat transform for vessels at different scales
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (scale, scale))
            tophat = cv2.morphologyEx((denoised * 255).astype(np.uint8), 
                                     cv2.MORPH_TOPHAT, kernel)
            vessel_responses.append(tophat / 255.0)
        
        # Combine multi-scale responses
        enhanced = np.maximum.reduce(vessel_responses)
        
        # Adaptive histogram equalization
        enhanced_uint8 = (enhanced * 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        enhanced_uint8 = clahe.apply(enhanced_uint8)
        
        return enhanced_uint8.astype(np.float64) / 255.0
    
    def process_dual_mode_optimized(self, iq_data: np.ndarray) -> Dict:
        """
        Optimized pipeline without any mixing operations
        
        Args:
            iq_data: Raw IQ data containing interleaved palmprint and palmvein
            
        Returns:
            Dictionary containing separated and reconstructed images
        """
        # Clear cache if needed
        self.cache_valid = False
        
        # Direct extraction and separation - no intermediate mixing
        palmprint_frames, palmvein_frames = self.extract_frames_direct(iq_data)
        
        # Parallel reconstruction for each modality
        palmprint_image = self.reconstruct_modality_optimized(
            palmprint_frames, modality_type='palmprint'
        )
        palmvein_image = self.reconstruct_modality_optimized(
            palmvein_frames, modality_type='palmvein'
        )
        
        # Compute quality metrics
        metrics = self.compute_separation_metrics(
            palmprint_image, palmvein_image, 
            len(palmprint_frames), len(palmvein_frames)
        )
        
        return {
            'palmprint_image': palmprint_image,
            'palmvein_image': palmvein_image,
            'frame_count': len(palmprint_frames) + len(palmvein_frames),
            'palmprint_frame_count': len(palmprint_frames),
            'palmvein_frame_count': len(palmvein_frames),
            'separation_metrics': metrics,
            'no_mixing': True  # Flag indicating no mixing was performed
        }
    
    def compute_separation_metrics(self, palmprint: np.ndarray, palmvein: np.ndarray,
                                  print_frames: int, vein_frames: int) -> Dict:
        """
        Compute quality metrics for separated modalities
        
        Args:
            palmprint: Reconstructed palmprint image
            palmvein: Reconstructed palmvein image
            print_frames: Number of palmprint frames
            vein_frames: Number of palmvein frames
            
        Returns:
            Dictionary of quality metrics
        """
        # Inter-modality difference (should be high for good separation)
        modality_difference = np.mean(np.abs(palmprint - palmvein))
        
        # Structural similarity difference
        palmprint_edges = cv2.Laplacian(palmprint.astype(np.float32), cv2.CV_32F).var()
        palmvein_edges = cv2.Laplacian(palmvein.astype(np.float32), cv2.CV_32F).var()
        
        # Frame balance (should be close to 1.0 for balanced capture)
        frame_balance = print_frames / (vein_frames + 1e-10)
        
        # Contrast metrics
        palmprint_contrast = np.std(palmprint)
        palmvein_contrast = np.std(palmvein)
        
        # SNR estimation for each modality
        palmprint_snr = self._estimate_snr(palmprint)
        palmvein_snr = self._estimate_snr(palmvein)
        
        return {
            'modality_difference': modality_difference,
            'palmprint_edge_strength': palmprint_edges,
            'palmvein_edge_strength': palmvein_edges,
            'frame_balance': frame_balance,
            'palmprint_contrast': palmprint_contrast,
            'palmvein_contrast': palmvein_contrast,
            'palmprint_snr': palmprint_snr,
            'palmvein_snr': palmvein_snr,
            'separation_quality': modality_difference * min(palmprint_contrast, palmvein_contrast)
        }
    
    def _estimate_snr(self, image: np.ndarray) -> float:
        """
        Estimate Signal-to-Noise Ratio (inspired by dsp.c)
        
        Args:
            image: Input image
            
        Returns:
            Estimated SNR in dB
        """
        # Smooth version as signal estimate
        signal = cv2.GaussianBlur(image, (5, 5), 1.0)
        
        # Difference as noise estimate
        noise = image - signal
        
        # Calculate SNR
        signal_power = np.var(signal)
        noise_power = np.var(noise)
        
        if noise_power > 0:
            snr_db = 10 * np.log10(signal_power / noise_power)
        else:
            snr_db = float('inf')
            
        return snr_db
    
    def adaptive_sync_correction(self, iq_data: np.ndarray, 
                                initial_boundaries: List[int]) -> List[int]:
        """
        Adaptive synchronization correction using correlation
        
        Args:
            iq_data: IQ data stream
            initial_boundaries: Initial frame boundary estimates
            
        Returns:
            Corrected frame boundaries
        """
        corrected_boundaries = [initial_boundaries[0]]
        
        for i in range(1, len(initial_boundaries)):
            expected_pos = initial_boundaries[i]
            
            # Search window around expected position
            search_window = int(self.frame_size * 0.1)
            start_search = max(0, expected_pos - search_window)
            end_search = min(len(iq_data), expected_pos + search_window)
            
            # Find actual boundary using local energy gradient
            search_region = np.abs(iq_data[start_search:end_search])
            gradient = np.gradient(search_region)
            
            # Find maximum gradient (frame transition)
            peak_idx = np.argmax(np.abs(gradient))
            actual_pos = start_search + peak_idx
            
            corrected_boundaries.append(actual_pos)
        
        return corrected_boundaries


# Utility functions for batch processing
class BatchProcessor:
    """
    Batch processing utilities for handling large datasets
    """
    
    @staticmethod
    def process_multiple_captures(iq_data_list: List[np.ndarray],
                                 frame_width: int, frame_height: int) -> Dict:
        """
        Process multiple IQ captures in batch
        
        Args:
            iq_data_list: List of IQ data captures
            frame_width: Frame width
            frame_height: Frame height
            
        Returns:
            Combined results from all captures
        """
        separator = OptimizedDualModeSeparation(frame_width, frame_height)
        
        all_palmprints = []
        all_palmveins = []
        all_metrics = []
        
        for iq_data in iq_data_list:
            result = separator.process_dual_mode_optimized(iq_data)
            all_palmprints.append(result['palmprint_image'])
            all_palmveins.append(result['palmvein_image'])
            all_metrics.append(result['separation_metrics'])
        
        # Average images across captures for noise reduction
        avg_palmprint = np.mean(all_palmprints, axis=0)
        avg_palmvein = np.mean(all_palmveins, axis=0)
        
        # Compute overall metrics
        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])
        
        return {
            'averaged_palmprint': avg_palmprint,
            'averaged_palmvein': avg_palmvein,
            'all_palmprints': all_palmprints,
            'all_palmveins': all_palmveins,
            'metrics': avg_metrics,
            'capture_count': len(iq_data_list)
        }