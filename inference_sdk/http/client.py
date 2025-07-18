from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Literal, Optional, Tuple, Union

import aiohttp
import numpy as np
import requests
from aiohttp import ClientConnectionError, ClientResponseError
from requests import HTTPError

from inference_sdk.http.entities import (
    ALL_ROBOFLOW_API_URLS,
    CLASSIFICATION_TASK,
    INSTANCE_SEGMENTATION_TASK,
    KEYPOINTS_DETECTION_TASK,
    OBJECT_DETECTION_TASK,
    HTTPClientMode,
    ImagesReference,
    InferenceConfiguration,
    ModelDescription,
    RegisteredModels,
    ServerInfo,
)
from inference_sdk.http.errors import (
    APIKeyNotProvided,
    HTTPCallErrorError,
    HTTPClientError,
    InvalidModelIdentifier,
    InvalidParameterError,
    ModelNotInitializedError,
    ModelNotSelectedError,
    ModelTaskTypeNotSupportedError,
    WrongClientModeError,
)
from inference_sdk.http.utils.aliases import (
    resolve_ocr_path,
    resolve_roboflow_model_alias,
)
from inference_sdk.http.utils.executors import (
    RequestMethod,
    execute_requests_packages,
    execute_requests_packages_async,
)
from inference_sdk.http.utils.iterables import unwrap_single_element_list
from inference_sdk.http.utils.loaders import (
    load_nested_batches_of_inference_input,
    load_static_inference_input,
    load_static_inference_input_async,
    load_stream_inference_input,
)
from inference_sdk.http.utils.post_processing import (
    adjust_prediction_to_client_scaling_factor,
    combine_clip_embeddings,
    combine_gaze_detections,
    decode_workflow_outputs,
    filter_model_descriptions,
    response_contains_jpeg_image,
    transform_base64_visualisation,
    transform_visualisation_bytes,
)
from inference_sdk.http.utils.profilling import save_workflows_profiler_trace
from inference_sdk.http.utils.request_building import (
    ImagePlacement,
    prepare_requests_data,
)
from inference_sdk.http.utils.requests import (
    api_key_safe_raise_for_status,
    deduct_api_key_from_string,
    inject_images_into_payload,
    inject_nested_batches_of_images_into_payload,
)
from inference_sdk.utils.decorators import deprecated, experimental

SUCCESSFUL_STATUS_CODE = 200
DEFAULT_HEADERS = {
    "Content-Type": "application/json",
}
NEW_INFERENCE_ENDPOINTS = {
    INSTANCE_SEGMENTATION_TASK: "/infer/instance_segmentation",
    OBJECT_DETECTION_TASK: "/infer/object_detection",
    CLASSIFICATION_TASK: "/infer/classification",
    KEYPOINTS_DETECTION_TASK: "/infer/keypoints_detection",
}
CLIP_ARGUMENT_TYPES = {"image", "text"}

BufferFillingStrategy = Literal[
    "WAIT", "DROP_OLDEST", "ADAPTIVE_DROP_OLDEST", "DROP_LATEST", "ADAPTIVE_DROP_LATEST"
]
BufferConsumptionStrategy = Literal["LAZY", "EAGER"]


def wrap_errors(function: callable) -> callable:
    def decorate(*args, **kwargs) -> Any:
        try:
            return function(*args, **kwargs)
        except HTTPError as error:
            if "application/json" in error.response.headers.get("Content-Type", ""):
                error_data = error.response.json()
                api_message = error_data.get("message", "N/A")
                if "inner_error_message" in error_data:
                    more_details = error_data["inner_error_message"]
                    api_message = f"{api_message}. More details: {more_details}"
            else:
                api_message = error.response.text
            raise HTTPCallErrorError(
                description=str(error),
                status_code=error.response.status_code,
                api_message=api_message,
            ) from error
        except ConnectionError as error:
            raise HTTPClientError(
                f"Error with server connection: {deduct_api_key_from_string(str(error))}"
            ) from error

    return decorate


def wrap_errors_async(function: callable) -> callable:
    async def decorate(*args, **kwargs) -> Any:
        try:
            return await function(*args, **kwargs)
        except ClientResponseError as error:
            raise HTTPCallErrorError(
                description=deduct_api_key_from_string(value=str(error)),
                status_code=error.status,
                api_message=deduct_api_key_from_string(error.message),
            ) from error
        except ClientConnectionError as error:
            raise HTTPClientError(
                f"Error with server connection: {deduct_api_key_from_string(str(error))}"
            ) from error

    return decorate


class InferenceHTTPClient:
    """HTTP client for making inference requests to Roboflow's API.

    This client handles authentication, request formatting, and error handling for
    interacting with Roboflow's inference endpoints. It supports both synchronous
    and asynchronous requests.

    Attributes:
        inference_configuration (InferenceConfiguration): Configuration settings for
            inference requests.
        client_mode (HTTPClientMode): The API version mode being used (V0 or V1).
        selected_model (Optional[str]): Currently selected model identifier, if any.

    Example:
        ```python
        from inference_sdk import InferenceHTTPClient

        client = InferenceHTTPClient(
            api_url="http://localhost:9001", # use local inference server
            # api_key="<YOUR API KEY>" # optional to access your private data and models
        )

        result = client.run_workflow(
            workspace_name="roboflow-docs",
            workflow_id="model-comparison",
            images={
                "image": "https://media.roboflow.com/workflows/examples/bleachers.jpg"
            },
            parameters={
                "model1": "yolov8n-640",
                "model2": "yolov11n-640"
            }
        )
        ```
    """

    @classmethod
    def init(
        cls,
        api_url: str,
        api_key: Optional[str] = None,
    ) -> "InferenceHTTPClient":
        """Initialize a new InferenceHTTPClient instance.

        Args:
            api_url (str): The base URL for the inference API.
            api_key (Optional[str], optional): API key for authentication. Defaults to None.

        Returns:
            InferenceHTTPClient: A new instance of the InferenceHTTPClient.
        """
        return cls(api_url=api_url, api_key=api_key)

    def __init__(
        self,
        api_url: str,
        api_key: Optional[str] = None,
    ):
        """Initialize a new InferenceHTTPClient instance.

        Args:
            api_url (str): The base URL for the inference API.
            api_key (Optional[str], optional): API key for authentication. Defaults to None.
        """
        self.__api_url = api_url
        self.__api_key = api_key
        self.__inference_configuration = InferenceConfiguration.init_default()
        self.__client_mode = _determine_client_mode(api_url=api_url)
        self.__selected_model: Optional[str] = None

    @property
    def inference_configuration(self) -> InferenceConfiguration:
        """Get the current inference configuration.

        Returns:
            InferenceConfiguration: The current inference configuration settings.
        """
        return self.__inference_configuration

    @property
    def client_mode(self) -> HTTPClientMode:
        """Get the current client mode.

        Returns:
            HTTPClientMode: The current API version mode (V0 or V1).
        """
        return self.__client_mode

    @property
    def selected_model(self) -> Optional[str]:
        """Get the currently selected model identifier.

        Returns:
            Optional[str]: The identifier of the currently selected model, if any.
        """
        return self.__selected_model

    @contextmanager
    def use_configuration(
        self, inference_configuration: InferenceConfiguration
    ) -> Generator["InferenceHTTPClient", None, None]:
        """Temporarily use a different inference configuration.

        Args:
            inference_configuration (InferenceConfiguration): The temporary configuration to use.

        Yields:
            Generator[InferenceHTTPClient, None, None]: The client instance with temporary configuration.
        """
        previous_configuration = self.__inference_configuration
        self.__inference_configuration = inference_configuration
        try:
            yield self
        finally:
            self.__inference_configuration = previous_configuration

    def configure(
        self, inference_configuration: InferenceConfiguration
    ) -> "InferenceHTTPClient":
        """Configure the client with new inference settings.

        Args:
            inference_configuration (InferenceConfiguration): The new configuration to apply.

        Returns:
            InferenceHTTPClient: The client instance with updated configuration.
        """
        self.__inference_configuration = inference_configuration
        return self

    def select_api_v0(self) -> "InferenceHTTPClient":
        """Select API version 0 for client operations.

        Returns:
            InferenceHTTPClient: The client instance with API v0 selected.
        """
        self.__client_mode = HTTPClientMode.V0
        return self

    def select_api_v1(self) -> "InferenceHTTPClient":
        """Select API version 1 for client operations.

        Returns:
            InferenceHTTPClient: The client instance with API v1 selected.
        """
        self.__client_mode = HTTPClientMode.V1
        return self

    @contextmanager
    def use_api_v0(self) -> Generator["InferenceHTTPClient", None, None]:
        """Temporarily use API version 0 for client operations.

        Yields:
            Generator[InferenceHTTPClient, None, None]: The client instance temporarily using API v0.
        """
        previous_client_mode = self.__client_mode
        self.__client_mode = HTTPClientMode.V0
        try:
            yield self
        finally:
            self.__client_mode = previous_client_mode

    @contextmanager
    def use_api_v1(self) -> Generator["InferenceHTTPClient", None, None]:
        """Temporarily use API version 1 for client operations.

        Yields:
            Generator[InferenceHTTPClient, None, None]: The client instance temporarily using API v1.
        """
        previous_client_mode = self.__client_mode
        self.__client_mode = HTTPClientMode.V1
        try:
            yield self
        finally:
            self.__client_mode = previous_client_mode

    def select_model(self, model_id: str) -> "InferenceHTTPClient":
        """Select a model for inference operations.

        Args:
            model_id (str): The identifier of the model to select.

        Returns:
            InferenceHTTPClient: The client instance with the selected model.
        """
        self.__selected_model = model_id
        return self

    @contextmanager
    def use_model(self, model_id: str) -> Generator["InferenceHTTPClient", None, None]:
        """Temporarily use a specific model for inference operations.

        Args:
            model_id (str): The identifier of the model to use.

        Yields:
            Generator[InferenceHTTPClient, None, None]: The client instance temporarily using the specified model.
        """
        previous_model = self.__selected_model
        self.__selected_model = model_id
        try:
            yield self
        finally:
            self.__selected_model = previous_model

    @wrap_errors
    def get_server_info(self) -> ServerInfo:
        """Get information about the inference server.

        Returns:
            ServerInfo: Information about the server configuration and status.

        Raises:
            HTTPCallErrorError: If there is an error in the HTTP call.
            HTTPClientError: If there is an error with the server connection.
        """
        response = requests.get(f"{self.__api_url}/info")
        response.raise_for_status()
        response_payload = response.json()
        return ServerInfo.from_dict(response_payload)

    def infer_on_stream(
        self,
        input_uri: str,
        model_id: Optional[str] = None,
    ) -> Generator[Tuple[Union[str, int], np.ndarray, dict], None, None]:
        """Run inference on a video stream or sequence of images.

        Args:
            input_uri (str): URI of the input stream or directory.
            model_id (Optional[str], optional): Model identifier to use for inference. Defaults to None.

        Yields:
            Generator[Tuple[Union[str, int], np.ndarray, dict], None, None]: Tuples of (frame reference, frame data, prediction).
        """
        for reference, frame in load_stream_inference_input(
            input_uri=input_uri,
            image_extensions=self.__inference_configuration.image_extensions_for_directory_scan,
        ):
            prediction = self.infer(
                inference_input=frame,
                model_id=model_id,
            )
            yield reference, frame, prediction

    @wrap_errors
    def infer(
        self,
        inference_input: Union[ImagesReference, List[ImagesReference]],
        model_id: Optional[str] = None,
    ) -> Union[dict, List[dict]]:
        """Run inference on one or more images.

        Args:
            inference_input (Union[ImagesReference, List[ImagesReference]]): Input image(s) for inference.
            model_id (Optional[str], optional): Model identifier to use for inference. Defaults to None.

        Returns:
            Union[dict, List[dict]]: Inference results for the input image(s).

        Raises:
            HTTPCallErrorError: If there is an error in the HTTP call.
            HTTPClientError: If there is an error with the server connection.
        """
        if self.__client_mode is HTTPClientMode.V0:
            return self.infer_from_api_v0(
                inference_input=inference_input,
                model_id=model_id,
            )
        return self.infer_from_api_v1(
            inference_input=inference_input,
            model_id=model_id,
        )

    @wrap_errors_async
    async def infer_async(
        self,
        inference_input: Union[ImagesReference, List[ImagesReference]],
        model_id: Optional[str] = None,
    ) -> Union[dict, List[dict]]:
        """Run inference asynchronously on one or more images.

        Args:
            inference_input (Union[ImagesReference, List[ImagesReference]]): Input image(s) for inference.
            model_id (Optional[str], optional): Model identifier to use for inference. Defaults to None.

        Returns:
            Union[dict, List[dict]]: Inference results for the input image(s).

        Raises:
            HTTPCallErrorError: If there is an error in the HTTP call.
            HTTPClientError: If there is an error with the server connection.
        """
        if self.__client_mode is HTTPClientMode.V0:
            return await self.infer_from_api_v0_async(
                inference_input=inference_input,
                model_id=model_id,
            )
        return await self.infer_from_api_v1_async(
            inference_input=inference_input,
            model_id=model_id,
        )

    def infer_from_api_v0(
        self,
        inference_input: Union[ImagesReference, List[ImagesReference]],
        model_id: Optional[str] = None,
    ) -> Union[dict, List[dict]]:
        """Run inference using API v0.

        Args:
            inference_input (Union[ImagesReference, List[ImagesReference]]): Input image(s) for inference.
            model_id (Optional[str], optional): Model identifier to use for inference. Defaults to None.

        Returns:
            Union[dict, List[dict]]: Inference results for the input image(s).

        Raises:
            ModelNotSelectedError: If no model is selected.
            APIKeyNotProvided: If API key is required but not provided.
            InvalidModelIdentifier: If the model identifier format is invalid.
        """
        model_id_to_be_used = model_id or self.__selected_model
        _ensure_model_is_selected(model_id=model_id_to_be_used)
        _ensure_api_key_provided(api_key=self.__api_key)
        model_id_to_be_used = resolve_roboflow_model_alias(model_id=model_id_to_be_used)
        model_id_chunks = model_id_to_be_used.split("/")
        if len(model_id_chunks) != 2:
            raise InvalidModelIdentifier(
                f"Invalid model id: {model_id}. Expected format: project_id/model_version_id."
            )
        max_height, max_width = _determine_client_downsizing_parameters(
            client_downsizing_disabled=self.__inference_configuration.client_downsizing_disabled,
            model_description=None,
            default_max_input_size=self.__inference_configuration.default_max_input_size,
        )
        encoded_inference_inputs = load_static_inference_input(
            inference_input=inference_input,
            max_height=max_height,
            max_width=max_width,
        )
        params = {
            "api_key": self.__api_key,
        }
        params.update(self.__inference_configuration.to_legacy_call_parameters())
        requests_data = prepare_requests_data(
            url=f"{self.__api_url}/{model_id_chunks[0]}/{model_id_chunks[1]}",
            encoded_inference_inputs=encoded_inference_inputs,
            headers=DEFAULT_HEADERS,
            parameters=params,
            payload=None,
            max_batch_size=1,
            image_placement=ImagePlacement.DATA,
        )
        responses = execute_requests_packages(
            requests_data=requests_data,
            request_method=RequestMethod.POST,
            max_concurrent_requests=self.__inference_configuration.max_concurrent_requests,
        )
        results = []
        for request_data, response in zip(requests_data, responses):
            if response_contains_jpeg_image(response=response):
                visualisation = transform_visualisation_bytes(
                    visualisation=response.content,
                    expected_format=self.__inference_configuration.output_visualisation_format,
                )
                parsed_response = {"visualization": visualisation}
            else:
                parsed_response = response.json()
                if parsed_response.get("visualization") is not None:
                    parsed_response["visualization"] = transform_base64_visualisation(
                        visualisation=parsed_response["visualization"],
                        expected_format=self.__inference_configuration.output_visualisation_format,
                    )
            parsed_response = adjust_prediction_to_client_scaling_factor(
                prediction=parsed_response,
                scaling_factor=request_data.image_scaling_factors[0],
            )
            results.append(parsed_response)
        return unwrap_single_element_list(sequence=results)

    async def infer_from_api_v0_async(
        self,
        inference_input: Union[ImagesReference, List[ImagesReference]],
        model_id: Optional[str] = None,
    ) -> Union[dict, List[dict]]:
        """Run inference using API v0 asynchronously.

        Args:
            inference_input (Union[ImagesReference, List[ImagesReference]]): Input image(s) for inference.
            model_id (Optional[str], optional): Model identifier to use for inference. Defaults to None.

        Returns:
            Union[dict, List[dict]]: Inference results for the input image(s).

        Raises:
            ModelNotSelectedError: If no model is selected.
            APIKeyNotProvided: If API key is required but not provided.
            InvalidModelIdentifier: If the model identifier format is invalid.
        """
        model_id_to_be_used = model_id or self.__selected_model
        _ensure_model_is_selected(model_id=model_id_to_be_used)
        _ensure_api_key_provided(api_key=self.__api_key)
        model_id_to_be_used = resolve_roboflow_model_alias(model_id=model_id_to_be_used)
        model_id_chunks = model_id_to_be_used.split("/")
        if len(model_id_chunks) != 2:
            raise InvalidModelIdentifier(
                f"Invalid model id: {model_id}. Expected format: project_id/model_version_id."
            )
        max_height, max_width = _determine_client_downsizing_parameters(
            client_downsizing_disabled=self.__inference_configuration.client_downsizing_disabled,
            model_description=None,
            default_max_input_size=self.__inference_configuration.default_max_input_size,
        )
        encoded_inference_inputs = await load_static_inference_input_async(
            inference_input=inference_input,
            max_height=max_height,
            max_width=max_width,
        )
        params = {
            "api_key": self.__api_key,
        }
        params.update(self.__inference_configuration.to_legacy_call_parameters())
        requests_data = prepare_requests_data(
            url=f"{self.__api_url}/{model_id_chunks[0]}/{model_id_chunks[1]}",
            encoded_inference_inputs=encoded_inference_inputs,
            headers=DEFAULT_HEADERS,
            parameters=params,
            payload=None,
            max_batch_size=1,
            image_placement=ImagePlacement.DATA,
        )
        responses = await execute_requests_packages_async(
            requests_data=requests_data,
            request_method=RequestMethod.POST,
            max_concurrent_requests=self.__inference_configuration.max_concurrent_requests,
        )
        results = []
        for request_data, response in zip(requests_data, responses):
            if not issubclass(type(response), dict):
                visualisation = transform_visualisation_bytes(
                    visualisation=response,
                    expected_format=self.__inference_configuration.output_visualisation_format,
                )
                parsed_response = {"visualization": visualisation}
            else:
                parsed_response = response
                if parsed_response.get("visualization") is not None:
                    parsed_response["visualization"] = transform_base64_visualisation(
                        visualisation=parsed_response["visualization"],
                        expected_format=self.__inference_configuration.output_visualisation_format,
                    )
            parsed_response = adjust_prediction_to_client_scaling_factor(
                prediction=parsed_response,
                scaling_factor=request_data.image_scaling_factors[0],
            )
            results.append(parsed_response)
        return unwrap_single_element_list(sequence=results)

    def infer_from_api_v1(
        self,
        inference_input: Union[ImagesReference, List[ImagesReference]],
        model_id: Optional[str] = None,
    ) -> Union[dict, List[dict]]:
        self.__ensure_v1_client_mode()
        model_id_to_be_used = model_id or self.__selected_model
        _ensure_model_is_selected(model_id=model_id_to_be_used)
        model_id_to_be_used = resolve_roboflow_model_alias(model_id=model_id_to_be_used)
        model_description = self.get_model_description(model_id=model_id_to_be_used)
        max_height, max_width = _determine_client_downsizing_parameters(
            client_downsizing_disabled=self.__inference_configuration.client_downsizing_disabled,
            model_description=model_description,
            default_max_input_size=self.__inference_configuration.default_max_input_size,
        )
        if model_description.task_type not in NEW_INFERENCE_ENDPOINTS:
            raise ModelTaskTypeNotSupportedError(
                f"Model task {model_description.task_type} is not supported by API v1 client."
            )
        encoded_inference_inputs = load_static_inference_input(
            inference_input=inference_input,
            max_height=max_height,
            max_width=max_width,
        )
        payload = {
            "api_key": self.__api_key,
            "model_id": model_id_to_be_used,
        }
        endpoint = NEW_INFERENCE_ENDPOINTS[model_description.task_type]
        payload.update(
            self.__inference_configuration.to_api_call_parameters(
                client_mode=self.__client_mode,
                task_type=model_description.task_type,
            )
        )
        requests_data = prepare_requests_data(
            url=f"{self.__api_url}{endpoint}",
            encoded_inference_inputs=encoded_inference_inputs,
            headers=DEFAULT_HEADERS,
            parameters=None,
            payload=payload,
            max_batch_size=self.__inference_configuration.max_batch_size,
            image_placement=ImagePlacement.JSON,
        )
        responses = execute_requests_packages(
            requests_data=requests_data,
            request_method=RequestMethod.POST,
            max_concurrent_requests=self.__inference_configuration.max_concurrent_requests,
        )
        results = []
        for request_data, response in zip(requests_data, responses):
            parsed_response = response.json()
            if not issubclass(type(parsed_response), list):
                parsed_response = [parsed_response]
            for parsed_response_element, scaling_factor in zip(
                parsed_response, request_data.image_scaling_factors
            ):
                if parsed_response_element.get("visualization") is not None:
                    parsed_response_element["visualization"] = (
                        transform_base64_visualisation(
                            visualisation=parsed_response_element["visualization"],
                            expected_format=self.__inference_configuration.output_visualisation_format,
                        )
                    )
                parsed_response_element = adjust_prediction_to_client_scaling_factor(
                    prediction=parsed_response_element,
                    scaling_factor=scaling_factor,
                )
                results.append(parsed_response_element)
        return unwrap_single_element_list(sequence=results)

    async def infer_from_api_v1_async(
        self,
        inference_input: Union[ImagesReference, List[ImagesReference]],
        model_id: Optional[str] = None,
    ) -> Union[dict, List[dict]]:
        self.__ensure_v1_client_mode()
        model_id_to_be_used = model_id or self.__selected_model
        _ensure_model_is_selected(model_id=model_id_to_be_used)
        model_id_to_be_used = resolve_roboflow_model_alias(model_id=model_id_to_be_used)
        model_description = await self.get_model_description_async(
            model_id=model_id_to_be_used
        )
        max_height, max_width = _determine_client_downsizing_parameters(
            client_downsizing_disabled=self.__inference_configuration.client_downsizing_disabled,
            model_description=model_description,
            default_max_input_size=self.__inference_configuration.default_max_input_size,
        )
        if model_description.task_type not in NEW_INFERENCE_ENDPOINTS:
            raise ModelTaskTypeNotSupportedError(
                f"Model task {model_description.task_type} is not supported by API v1 client."
            )
        encoded_inference_inputs = await load_static_inference_input_async(
            inference_input=inference_input,
            max_height=max_height,
            max_width=max_width,
        )
        payload = {
            "api_key": self.__api_key,
            "model_id": model_id_to_be_used,
        }
        endpoint = NEW_INFERENCE_ENDPOINTS[model_description.task_type]
        payload.update(
            self.__inference_configuration.to_api_call_parameters(
                client_mode=self.__client_mode,
                task_type=model_description.task_type,
            )
        )
        requests_data = prepare_requests_data(
            url=f"{self.__api_url}{endpoint}",
            encoded_inference_inputs=encoded_inference_inputs,
            headers=DEFAULT_HEADERS,
            parameters=None,
            payload=payload,
            max_batch_size=self.__inference_configuration.max_batch_size,
            image_placement=ImagePlacement.JSON,
        )
        responses = await execute_requests_packages_async(
            requests_data=requests_data,
            request_method=RequestMethod.POST,
            max_concurrent_requests=self.__inference_configuration.max_concurrent_requests,
        )
        results = []
        for request_data, parsed_response in zip(requests_data, responses):
            if not issubclass(type(parsed_response), list):
                parsed_response = [parsed_response]
            for parsed_response_element, scaling_factor in zip(
                parsed_response, request_data.image_scaling_factors
            ):
                if parsed_response_element.get("visualization") is not None:
                    parsed_response_element["visualization"] = (
                        transform_base64_visualisation(
                            visualisation=parsed_response_element["visualization"],
                            expected_format=self.__inference_configuration.output_visualisation_format,
                        )
                    )
                parsed_response_element = adjust_prediction_to_client_scaling_factor(
                    prediction=parsed_response_element,
                    scaling_factor=scaling_factor,
                )
                results.append(parsed_response_element)
        return unwrap_single_element_list(sequence=results)

    def get_model_description(
        self, model_id: str, allow_loading: bool = True
    ) -> ModelDescription:
        """Get the description of a model.

        Args:
            model_id (str): The identifier of the model.
            allow_loading (bool, optional): Whether to load the model if not already loaded. Defaults to True.

        Returns:
            ModelDescription: Description of the model.

        Raises:
            WrongClientModeError: If not in API v1 mode.
            ModelNotInitializedError: If the model is not initialized and cannot be loaded.
        """
        self.__ensure_v1_client_mode()
        de_aliased_model_id = resolve_roboflow_model_alias(model_id=model_id)
        registered_models = self.list_loaded_models()
        matching_model = filter_model_descriptions(
            descriptions=registered_models.models,
            model_id=de_aliased_model_id,
        )
        if matching_model is None and allow_loading is True:
            registered_models = self.load_model(model_id=de_aliased_model_id)
            matching_model = filter_model_descriptions(
                descriptions=registered_models.models,
                model_id=de_aliased_model_id,
            )
        if matching_model is not None:
            return matching_model
        raise ModelNotInitializedError(
            f"Model {model_id} (de-aliased: {de_aliased_model_id}) is not initialised and cannot "
            f"retrieve its description."
        )

    async def get_model_description_async(
        self, model_id: str, allow_loading: bool = True
    ) -> ModelDescription:
        """Get the description of a model asynchronously.

        Args:
            model_id (str): The identifier of the model.
            allow_loading (bool, optional): Whether to load the model if not already loaded. Defaults to True.

        Returns:
            ModelDescription: Description of the model.

        Raises:
            WrongClientModeError: If not in API v1 mode.
            ModelNotInitializedError: If the model is not initialized and cannot be loaded.
        """
        self.__ensure_v1_client_mode()
        de_aliased_model_id = resolve_roboflow_model_alias(model_id=model_id)
        registered_models = await self.list_loaded_models_async()
        matching_model = filter_model_descriptions(
            descriptions=registered_models.models,
            model_id=de_aliased_model_id,
        )
        if matching_model is None and allow_loading is True:
            registered_models = await self.load_model_async(
                model_id=de_aliased_model_id
            )
            matching_model = filter_model_descriptions(
                descriptions=registered_models.models,
                model_id=de_aliased_model_id,
            )
        if matching_model is not None:
            return matching_model
        raise ModelNotInitializedError(
            f"Model {model_id} (de-aliased: {de_aliased_model_id}) is not initialised and cannot "
            f"retrieve its description."
        )

    @wrap_errors
    def list_loaded_models(self) -> RegisteredModels:
        """List all models currently loaded on the server.

        Returns:
            RegisteredModels: Information about registered models.

        Raises:
            WrongClientModeError: If not in API v1 mode.
            HTTPCallErrorError: If there is an error in the HTTP call.
            HTTPClientError: If there is an error with the server connection.
        """
        self.__ensure_v1_client_mode()
        response = requests.get(
            f"{self.__api_url}/model/registry?api_key={self.__api_key}"
        )
        response.raise_for_status()
        response_payload = response.json()
        return RegisteredModels.from_dict(response_payload)

    @wrap_errors_async
    async def list_loaded_models_async(self) -> RegisteredModels:
        """List all models currently loaded on the server asynchronously.

        Returns:
            RegisteredModels: Information about registered models.

        Raises:
            WrongClientModeError: If not in API v1 mode.
            HTTPCallErrorError: If there is an error in the HTTP call.
            HTTPClientError: If there is an error with the server connection.
        """
        self.__ensure_v1_client_mode()
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.__api_url}/model/registry?api_key={self.__api_key}"
            ) as response:
                response.raise_for_status()
                response_payload = await response.json()
                return RegisteredModels.from_dict(response_payload)

    @wrap_errors
    def load_model(
        self, model_id: str, set_as_default: bool = False
    ) -> RegisteredModels:
        """Load a model onto the server.

        Args:
            model_id (str): The identifier of the model to load.
            set_as_default (bool, optional): Whether to set this model as the default. Defaults to False.

        Returns:
            RegisteredModels: Updated information about registered models.

        Raises:
            WrongClientModeError: If not in API v1 mode.
            HTTPCallErrorError: If there is an error in the HTTP call.
            HTTPClientError: If there is an error with the server connection.
        """
        self.__ensure_v1_client_mode()
        de_aliased_model_id = resolve_roboflow_model_alias(model_id=model_id)
        response = requests.post(
            f"{self.__api_url}/model/add",
            json={
                "model_id": de_aliased_model_id,
                "api_key": self.__api_key,
            },
            headers=DEFAULT_HEADERS,
        )
        response.raise_for_status()
        response_payload = response.json()
        if set_as_default:
            self.__selected_model = de_aliased_model_id
        return RegisteredModels.from_dict(response_payload)

    @wrap_errors_async
    async def load_model_async(
        self, model_id: str, set_as_default: bool = False
    ) -> RegisteredModels:
        """Load a model onto the server asynchronously.

        Args:
            model_id (str): The identifier of the model to load.
            set_as_default (bool, optional): Whether to set this model as the default. Defaults to False.

        Returns:
            RegisteredModels: Updated information about registered models.

        Raises:
            WrongClientModeError: If not in API v1 mode.
            HTTPCallErrorError: If there is an error in the HTTP call.
            HTTPClientError: If there is an error with the server connection.
        """
        self.__ensure_v1_client_mode()
        de_aliased_model_id = resolve_roboflow_model_alias(model_id=model_id)
        payload = {
            "model_id": de_aliased_model_id,
            "api_key": self.__api_key,
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.__api_url}/model/add",
                json=payload,
                headers=DEFAULT_HEADERS,
            ) as response:
                response.raise_for_status()
                response_payload = await response.json()
        if set_as_default:
            self.__selected_model = de_aliased_model_id
        return RegisteredModels.from_dict(response_payload)

    @wrap_errors
    def unload_model(self, model_id: str) -> RegisteredModels:
        """Unload a model from the server.

        Args:
            model_id (str): The identifier of the model to unload.

        Returns:
            RegisteredModels: Updated information about registered models.

        Raises:
            WrongClientModeError: If not in API v1 mode.
            HTTPCallErrorError: If there is an error in the HTTP call.
            HTTPClientError: If there is an error with the server connection.
        """
        self.__ensure_v1_client_mode()
        de_aliased_model_id = resolve_roboflow_model_alias(model_id=model_id)
        response = requests.post(
            f"{self.__api_url}/model/remove",
            json={
                "model_id": de_aliased_model_id,
            },
            headers=DEFAULT_HEADERS,
        )
        response.raise_for_status()
        response_payload = response.json()
        if (
            de_aliased_model_id == self.__selected_model
            or model_id == self.__selected_model
        ):
            self.__selected_model = None
        return RegisteredModels.from_dict(response_payload)

    @wrap_errors_async
    async def unload_model_async(self, model_id: str) -> RegisteredModels:
        self.__ensure_v1_client_mode()
        de_aliased_model_id = resolve_roboflow_model_alias(model_id=model_id)
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.__api_url}/model/remove",
                json={
                    "model_id": de_aliased_model_id,
                },
                headers=DEFAULT_HEADERS,
            ) as response:
                response.raise_for_status()
                response_payload = await response.json()
        if (
            de_aliased_model_id == self.__selected_model
            or model_id == self.__selected_model
        ):
            self.__selected_model = None
        return RegisteredModels.from_dict(response_payload)

    @wrap_errors
    def unload_all_models(self) -> RegisteredModels:
        self.__ensure_v1_client_mode()
        response = requests.post(f"{self.__api_url}/model/clear")
        response.raise_for_status()
        response_payload = response.json()
        self.__selected_model = None
        return RegisteredModels.from_dict(response_payload)

    @wrap_errors_async
    async def unload_all_models_async(self) -> RegisteredModels:
        self.__ensure_v1_client_mode()
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.__api_url}/model/clear") as response:
                response.raise_for_status()
                response_payload = await response.json()
        self.__selected_model = None
        return RegisteredModels.from_dict(response_payload)

    @wrap_errors
    def ocr_image(
        self,
        inference_input: Union[ImagesReference, List[ImagesReference]],
        model: str = "doctr",
        version: Optional[str] = None,
    ) -> Union[dict, List[dict]]:
        """Run OCR on input image(s).

        Args:
            inference_input (Union[ImagesReference, List[ImagesReference]]): Input image(s) for OCR.
            model (str, optional): OCR model to use ('doctr' or 'trocr'). Defaults to "doctr".
            version (Optional[str], optional): Model version to use. Defaults to None.
                For trocr, supported versions are: 'trocr-small-printed', 'trocr-base-printed', 'trocr-large-printed'.

        Returns:
            Union[dict, List[dict]]: OCR results for the input image(s).

        Raises:
            HTTPCallErrorError: If there is an error in the HTTP call.
            HTTPClientError: If there is an error with the server connection.
        """
        encoded_inference_inputs = load_static_inference_input(
            inference_input=inference_input,
        )
        payload = self.__initialise_payload()
        if version:
            key = f"{model.lower()}_version_id"
            payload[key] = version
        model_path = resolve_ocr_path(model_name=model)
        url = self.__wrap_url_with_api_key(f"{self.__api_url}{model_path}")
        requests_data = prepare_requests_data(
            url=url,
            encoded_inference_inputs=encoded_inference_inputs,
            headers=DEFAULT_HEADERS,
            parameters=None,
            payload=payload,
            max_batch_size=1,
            image_placement=ImagePlacement.JSON,
        )
        responses = execute_requests_packages(
            requests_data=requests_data,
            request_method=RequestMethod.POST,
            max_concurrent_requests=self.__inference_configuration.max_concurrent_requests,
        )
        results = [r.json() for r in responses]
        return unwrap_single_element_list(sequence=results)

    @wrap_errors_async
    async def ocr_image_async(
        self,
        inference_input: Union[ImagesReference, List[ImagesReference]],
        model: str = "doctr",
        version: Optional[str] = None,
    ) -> Union[dict, List[dict]]:
        """Run OCR on input image(s) asynchronously.

        Args:
            inference_input (Union[ImagesReference, List[ImagesReference]]): Input image(s) for OCR.
            model (str, optional): OCR model to use ('doctr' or 'trocr'). Defaults to "doctr".
            version (Optional[str], optional): Model version to use. Defaults to None.
                For trocr, supported versions are: 'trocr-small-printed', 'trocr-base-printed', 'trocr-large-printed'.

        Returns:
            Union[dict, List[dict]]: OCR results for the input image(s).

        Raises:
            HTTPCallErrorError: If there is an error in the HTTP call.
            HTTPClientError: If there is an error with the server connection.
        """
        encoded_inference_inputs = await load_static_inference_input_async(
            inference_input=inference_input,
        )
        payload = self.__initialise_payload()
        if version:
            key = f"{model.lower()}_version_id"
            payload[key] = version
        model_path = resolve_ocr_path(model_name=model)
        url = self.__wrap_url_with_api_key(f"{self.__api_url}{model_path}")
        requests_data = prepare_requests_data(
            url=url,
            encoded_inference_inputs=encoded_inference_inputs,
            headers=DEFAULT_HEADERS,
            parameters=None,
            payload=payload,
            max_batch_size=1,
            image_placement=ImagePlacement.JSON,
        )
        responses = await execute_requests_packages_async(
            requests_data=requests_data,
            request_method=RequestMethod.POST,
            max_concurrent_requests=self.__inference_configuration.max_concurrent_requests,
        )
        return unwrap_single_element_list(sequence=responses)

    @wrap_errors
    def detect_gazes(
        self,
        inference_input: Union[ImagesReference, List[ImagesReference]],
    ) -> Union[dict, List[dict]]:
        """Detect gazes in input image(s).

        Args:
            inference_input (Union[ImagesReference, List[ImagesReference]]): Input image(s) for gaze detection.

        Returns:
            Union[dict, List[dict]]: Gaze detection results for the input image(s).

        Raises:
            WrongClientModeError: If not in API v1 mode.
            HTTPCallErrorError: If there is an error in the HTTP call.
            HTTPClientError: If there is an error with the server connection.
        """
        self.__ensure_v1_client_mode()  # Lambda does not support Gaze, so we require v1 mode of client
        result = self._post_images(
            inference_input=inference_input, endpoint="/gaze/gaze_detection"
        )
        return combine_gaze_detections(detections=result)

    @wrap_errors_async
    async def detect_gazes_async(
        self,
        inference_input: Union[ImagesReference, List[ImagesReference]],
    ) -> Union[dict, List[dict]]:
        """Detect gazes in input image(s) asynchronously.

        Args:
            inference_input (Union[ImagesReference, List[ImagesReference]]): Input image(s) for gaze detection.

        Returns:
            Union[dict, List[dict]]: Gaze detection results for the input image(s).

        Raises:
            WrongClientModeError: If not in API v1 mode.
            HTTPCallErrorError: If there is an error in the HTTP call.
            HTTPClientError: If there is an error with the server connection.
        """
        self.__ensure_v1_client_mode()  # Lambda does not support Gaze, so we require v1 mode of client
        result = await self._post_images_async(
            inference_input=inference_input, endpoint="/gaze/gaze_detection"
        )
        return combine_gaze_detections(detections=result)

    @wrap_errors
    def get_clip_image_embeddings(
        self,
        inference_input: Union[ImagesReference, List[ImagesReference]],
        clip_version: Optional[str] = None,
    ) -> Union[dict, List[dict]]:
        """Get CLIP embeddings for input image(s).

        Args:
            inference_input (Union[ImagesReference, List[ImagesReference]]): Input image(s) to embed.
            clip_version (Optional[str], optional): Version of CLIP model to use. Defaults to None.

        Returns:
            Union[dict, List[dict]]: CLIP embeddings for the input image(s).

        Raises:
            HTTPCallErrorError: If there is an error in the HTTP call.
            HTTPClientError: If there is an error with the server connection.
        """
        extra_payload = {}
        if clip_version is not None:
            extra_payload["clip_version_id"] = clip_version
        result = self._post_images(
            inference_input=inference_input,
            endpoint="/clip/embed_image",
            extra_payload=extra_payload,
        )
        result = combine_clip_embeddings(embeddings=result)
        return unwrap_single_element_list(result)

    @wrap_errors_async
    async def get_clip_image_embeddings_async(
        self,
        inference_input: Union[ImagesReference, List[ImagesReference]],
        clip_version: Optional[str] = None,
    ) -> Union[dict, List[dict]]:
        """Get CLIP embeddings for input image(s) asynchronously.

        Args:
            inference_input (Union[ImagesReference, List[ImagesReference]]): Input image(s) to embed.
            clip_version (Optional[str], optional): Version of CLIP model to use. Defaults to None.

        Returns:
            Union[dict, List[dict]]: CLIP embeddings for the input image(s).

        Raises:
            HTTPCallErrorError: If there is an error in the HTTP call.
            HTTPClientError: If there is an error with the server connection.
        """
        extra_payload = {}
        if clip_version is not None:
            extra_payload["clip_version_id"] = clip_version
        result = await self._post_images_async(
            inference_input=inference_input,
            endpoint="/clip/embed_image",
            extra_payload=extra_payload,
        )
        result = combine_clip_embeddings(embeddings=result)
        return unwrap_single_element_list(result)

    @wrap_errors
    def get_clip_text_embeddings(
        self,
        text: Union[str, List[str]],
        clip_version: Optional[str] = None,
    ) -> Union[dict, List[dict]]:
        """Get CLIP embeddings for input text(s).

        Args:
            text (Union[str, List[str]]): Input text(s) to embed.
            clip_version (Optional[str], optional): Version of CLIP model to use. Defaults to None.

        Returns:
            Union[dict, List[dict]]: CLIP embeddings for the input text(s).

        Raises:
            HTTPCallErrorError: If there is an error in the HTTP call.
            HTTPClientError: If there is an error with the server connection.
        """
        payload = self.__initialise_payload()
        payload["text"] = text
        if clip_version is not None:
            payload["clip_version_id"] = clip_version
        response = requests.post(
            self.__wrap_url_with_api_key(f"{self.__api_url}/clip/embed_text"),
            json=payload,
            headers=DEFAULT_HEADERS,
        )
        api_key_safe_raise_for_status(response=response)
        return unwrap_single_element_list(sequence=response.json())

    @wrap_errors_async
    async def get_clip_text_embeddings_async(
        self,
        text: Union[str, List[str]],
        clip_version: Optional[str] = None,
    ) -> Union[dict, List[dict]]:
        """Get CLIP embeddings for input text(s) asynchronously.

        Args:
            text (Union[str, List[str]]): Input text(s) to embed.
            clip_version (Optional[str], optional): Version of CLIP model to use. Defaults to None.

        Returns:
            Union[dict, List[dict]]: CLIP embeddings for the input text(s).

        Raises:
            HTTPCallErrorError: If there is an error in the HTTP call.
            HTTPClientError: If there is an error with the server connection.
        """
        payload = self.__initialise_payload()
        payload["text"] = text
        if clip_version is not None:
            payload["clip_version_id"] = clip_version
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.__wrap_url_with_api_key(f"{self.__api_url}/clip/embed_text"),
                json=payload,
                headers=DEFAULT_HEADERS,
            ) as response:
                response.raise_for_status()
                response_payload = await response.json()
        return unwrap_single_element_list(sequence=response_payload)

    @wrap_errors
    def clip_compare(
        self,
        subject: Union[str, ImagesReference],
        prompt: Union[str, List[str], ImagesReference, List[ImagesReference]],
        subject_type: str = "image",
        prompt_type: str = "text",
        clip_version: Optional[str] = None,
    ) -> Union[dict, List[dict]]:
        """Compare a subject against prompts using CLIP embeddings.

        Args:
            subject (Union[str, ImagesReference]): The subject to compare (image or text).
            prompt (Union[str, List[str], ImagesReference, List[ImagesReference]]): The prompt(s) to compare against.
            subject_type (str, optional): Type of subject ('image' or 'text'). Defaults to "image".
            prompt_type (str, optional): Type of prompt(s) ('image' or 'text'). Defaults to "text".
            clip_version (Optional[str], optional): Version of CLIP model to use. Defaults to None.

        Returns:
            Union[dict, List[dict]]: Comparison results between subject and prompt(s).

        Raises:
            InvalidParameterError: If subject_type or prompt_type is invalid.
            HTTPCallErrorError: If there is an error in the HTTP call.
            HTTPClientError: If there is an error with the server connection.
        """
        if (
            subject_type not in CLIP_ARGUMENT_TYPES
            or prompt_type not in CLIP_ARGUMENT_TYPES
        ):
            raise InvalidParameterError(
                f"Could not accept `subject_type` and `prompt_type` with values different than {CLIP_ARGUMENT_TYPES}"
            )
        payload = self.__initialise_payload()
        payload["subject_type"] = subject_type
        payload["prompt_type"] = prompt_type
        if clip_version is not None:
            payload["clip_version_id"] = clip_version
        if subject_type == "image":
            encoded_image = load_static_inference_input(
                inference_input=subject,
            )
            payload = inject_images_into_payload(
                payload=payload, encoded_images=encoded_image, key="subject"
            )
        else:
            payload["subject"] = subject
        if prompt_type == "image":
            encoded_inference_inputs = load_static_inference_input(
                inference_input=prompt,
            )
            payload = inject_images_into_payload(
                payload=payload, encoded_images=encoded_inference_inputs, key="prompt"
            )
        else:
            payload["prompt"] = prompt
        response = requests.post(
            self.__wrap_url_with_api_key(f"{self.__api_url}/clip/compare"),
            json=payload,
            headers=DEFAULT_HEADERS,
        )
        api_key_safe_raise_for_status(response=response)
        return response.json()

    @wrap_errors_async
    async def clip_compare_async(
        self,
        subject: Union[str, ImagesReference],
        prompt: Union[str, List[str], ImagesReference, List[ImagesReference]],
        subject_type: str = "image",
        prompt_type: str = "text",
        clip_version: Optional[str] = None,
    ) -> Union[dict, List[dict]]:
        """Compare a subject against prompts using CLIP embeddings asynchronously.

        Args:
            subject (Union[str, ImagesReference]): The subject to compare (image or text).
            prompt (Union[str, List[str], ImagesReference, List[ImagesReference]]): The prompt(s) to compare against.
            subject_type (str, optional): Type of subject ('image' or 'text'). Defaults to "image".
            prompt_type (str, optional): Type of prompt(s) ('image' or 'text'). Defaults to "text".
            clip_version (Optional[str], optional): Version of CLIP model to use. Defaults to None.

        Returns:
            Union[dict, List[dict]]: Comparison results between subject and prompt(s).

        Raises:
            InvalidParameterError: If subject_type or prompt_type is invalid.
            HTTPCallErrorError: If there is an error in the HTTP call.
            HTTPClientError: If there is an error with the server connection.
        """
        if (
            subject_type not in CLIP_ARGUMENT_TYPES
            or prompt_type not in CLIP_ARGUMENT_TYPES
        ):
            raise InvalidParameterError(
                f"Could not accept `subject_type` and `prompt_type` with values different than {CLIP_ARGUMENT_TYPES}"
            )
        payload = self.__initialise_payload()
        payload["subject_type"] = subject_type
        payload["prompt_type"] = prompt_type
        if clip_version is not None:
            payload["clip_version_id"] = clip_version
        if subject_type == "image":
            encoded_image = await load_static_inference_input_async(
                inference_input=subject,
            )
            payload = inject_images_into_payload(
                payload=payload, encoded_images=encoded_image, key="subject"
            )
        else:
            payload["subject"] = subject
        if prompt_type == "image":
            encoded_inference_inputs = await load_static_inference_input_async(
                inference_input=prompt,
            )
            payload = inject_images_into_payload(
                payload=payload, encoded_images=encoded_inference_inputs, key="prompt"
            )
        else:
            payload["prompt"] = prompt

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.__wrap_url_with_api_key(f"{self.__api_url}/clip/compare"),
                json=payload,
                headers=DEFAULT_HEADERS,
            ) as response:
                response.raise_for_status()
                return await response.json()

    @wrap_errors
    def get_perception_encoder_image_embeddings(
        self,
        inference_input: Union[ImagesReference, List[ImagesReference]],
        perception_encoder_version: Optional[str] = None,
    ) -> Union[dict, List[dict]]:
        """Get Perception Encoder embeddings for input image(s)."""
        extra_payload = {}
        if perception_encoder_version is not None:
            extra_payload["perception_encoder_version_id"] = perception_encoder_version
        result = self._post_images(
            inference_input=inference_input,
            endpoint="/perception_encoder/embed_image",
            extra_payload=extra_payload,
        )
        return unwrap_single_element_list(result)

    @wrap_errors
    def get_perception_encoder_text_embeddings(
        self,
        text: Union[str, List[str]],
        perception_encoder_version: Optional[str] = None,
    ) -> Union[dict, List[dict]]:
        """Get Perception Encoder embeddings for input text(s)."""
        payload = self.__initialise_payload()
        payload["text"] = text
        if perception_encoder_version is not None:
            payload["perception_encoder_version_id"] = perception_encoder_version
        response = requests.post(
            self.__wrap_url_with_api_key(
                f"{self.__api_url}/perception_encoder/embed_text"
            ),
            json=payload,
            headers=DEFAULT_HEADERS,
        )
        api_key_safe_raise_for_status(response=response)
        return unwrap_single_element_list(sequence=response.json())

    @deprecated(
        reason="Please use run_workflow(...) method. This method will be removed end of Q2 2024"
    )
    @wrap_errors
    def infer_from_workflow(
        self,
        workspace_name: Optional[str] = None,
        workflow_name: Optional[str] = None,
        specification: Optional[dict] = None,
        images: Optional[Dict[str, Any]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        excluded_fields: Optional[List[str]] = None,
        use_cache: bool = True,
        enable_profiling: bool = False,
    ) -> List[Dict[str, Any]]:
        """Run inference using a workflow specification.

        Triggers inference from workflow specification at the inference HTTP
        side. Either (`workspace_name` and `workflow_name`) or `workflow_specification` must be
        provided. In the first case - definition of workflow will be fetched
        from Roboflow API, in the latter - `workflow_specification` will be
        used. `images` and `parameters` will be merged into workflow inputs,
        the distinction is made to make sure the SDK can easily serialise
        images and prepare a proper payload. Supported images are numpy arrays,
        PIL.Image and base64 images, links to images and local paths.
        `excluded_fields` will be added to request to filter out results
        of workflow execution at the server side.

        Args:
            workspace_name (Optional[str], optional): Name of the workspace containing the workflow. Defaults to None.
            workflow_name (Optional[str], optional): Name of the workflow. Defaults to None.
            specification (Optional[dict], optional): Direct workflow specification. Defaults to None.
            images (Optional[Dict[str, Any]], optional): Images to process. Defaults to None.
            parameters (Optional[Dict[str, Any]], optional): Additional parameters for the workflow. Defaults to None.
            excluded_fields (Optional[List[str]], optional): Fields to exclude from results. Defaults to None.
            use_cache (bool, optional): Whether to use cached results. Defaults to True.
            enable_profiling (bool, optional): Whether to enable profiling. Defaults to False.

        Returns:
            List[Dict[str, Any]]: Results of the workflow execution.

        Raises:
            InvalidParameterError: If neither workflow identifiers nor specification is provided.
            HTTPCallErrorError: If there is an error in the HTTP call.
            HTTPClientError: If there is an error with the server connection.
        """
        return self._run_workflow(
            workspace_name=workspace_name,
            workflow_id=workflow_name,
            specification=specification,
            images=images,
            parameters=parameters,
            excluded_fields=excluded_fields,
            legacy_endpoints=True,
            use_cache=use_cache,
            enable_profiling=enable_profiling,
        )

    @wrap_errors
    def run_workflow(
        self,
        workspace_name: Optional[str] = None,
        workflow_id: Optional[str] = None,
        specification: Optional[dict] = None,
        images: Optional[Dict[str, Any]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        excluded_fields: Optional[List[str]] = None,
        use_cache: bool = True,
        enable_profiling: bool = False,
    ) -> List[Dict[str, Any]]:
        """Run inference using a workflow specification.

        Triggers inference from workflow specification at the inference HTTP
        side. Either (`workspace_name` and `workflow_id`) or `workflow_specification` must be
        provided. In the first case - definition of workflow will be fetched
        from Roboflow API, in the latter - `workflow_specification` will be
        used. `images` and `parameters` will be merged into workflow inputs,
        the distinction is made to make sure the SDK can easily serialise
        images and prepare a proper payload. Supported images are numpy arrays,
        PIL.Image and base64 images, links to images and local paths.
        `excluded_fields` will be added to request to filter out results
        of workflow execution at the server side.

        **Important!**
        Method is not compatible with inference server <=0.9.18. Please migrate to newer version of
        the server before end of Q2 2024. Until that is done - use old method: infer_from_workflow(...).

        Note:
            Method is not compatible with inference server <=0.9.18. Please migrate to newer version of
            the server before end of Q2 2024. Until that is done - use old method: infer_from_workflow(...).

        Args:
            workspace_name (Optional[str], optional): Name of the workspace containing the workflow. Defaults to None.
            workflow_id (Optional[str], optional): ID of the workflow. Defaults to None.
            specification (Optional[dict], optional): Direct workflow specification. Defaults to None.
            images (Optional[Dict[str, Any]], optional): Images to process. Defaults to None.
            parameters (Optional[Dict[str, Any]], optional): Additional parameters for the workflow. Defaults to None.
            excluded_fields (Optional[List[str]], optional): Fields to exclude from results. Defaults to None.
            use_cache (bool, optional): Whether to use cached results. Defaults to True.
            enable_profiling (bool, optional): Whether to enable profiling. Defaults to False.

        Returns:
            List[Dict[str, Any]]: Results of the workflow execution.

        Raises:
            InvalidParameterError: If neither workflow identifiers nor specification is provided.
            HTTPCallErrorError: If there is an error in the HTTP call.
            HTTPClientError: If there is an error with the server connection.
        """
        return self._run_workflow(
            workspace_name=workspace_name,
            workflow_id=workflow_id,
            specification=specification,
            images=images,
            parameters=parameters,
            excluded_fields=excluded_fields,
            legacy_endpoints=False,
            use_cache=use_cache,
            enable_profiling=enable_profiling,
        )

    def _run_workflow(
        self,
        workspace_name: Optional[str] = None,
        workflow_id: Optional[str] = None,
        specification: Optional[dict] = None,
        images: Optional[Dict[str, Any]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        excluded_fields: Optional[List[str]] = None,
        legacy_endpoints: bool = False,
        use_cache: bool = True,
        enable_profiling: bool = False,
    ) -> List[Dict[str, Any]]:
        named_workflow_specified = (workspace_name is not None) and (
            workflow_id is not None
        )
        if not (named_workflow_specified != (specification is not None)):
            raise InvalidParameterError(
                "Parameters (`workspace_name`, `workflow_id` / `workflow_name`) can be used mutually exclusive with "
                "`specification`, but at least one must be set."
            )
        if images is None:
            images = {}
        if parameters is None:
            parameters = {}
        payload = {
            "api_key": self.__api_key,
            "use_cache": use_cache,
            "enable_profiling": enable_profiling,
        }
        inputs = {}
        for image_name, image in images.items():
            loaded_image = load_nested_batches_of_inference_input(
                inference_input=image,
            )
            inject_nested_batches_of_images_into_payload(
                payload=inputs,
                encoded_images=loaded_image,
                key=image_name,
            )
        inputs.update(parameters)
        payload["inputs"] = inputs
        if excluded_fields is not None:
            payload["excluded_fields"] = excluded_fields
        if specification is not None:
            payload["specification"] = specification
        if specification is not None:
            if legacy_endpoints:
                url = f"{self.__api_url}/infer/workflows"
            else:
                url = f"{self.__api_url}/workflows/run"
        else:
            if legacy_endpoints:
                url = f"{self.__api_url}/infer/workflows/{workspace_name}/{workflow_id}"
            else:
                url = f"{self.__api_url}/{workspace_name}/workflows/{workflow_id}"
        response = requests.post(
            url,
            json=payload,
            headers=DEFAULT_HEADERS,
        )
        api_key_safe_raise_for_status(response=response)
        response_data = response.json()
        workflow_outputs = response_data["outputs"]
        profiler_trace = response_data.get("profiler_trace", [])
        if enable_profiling:
            save_workflows_profiler_trace(
                directory=self.__inference_configuration.profiling_directory,
                profiler_trace=profiler_trace,
            )
        return decode_workflow_outputs(
            workflow_outputs=workflow_outputs,
            expected_format=self.__inference_configuration.output_visualisation_format,
        )

    @wrap_errors
    def infer_from_yolo_world(
        self,
        inference_input: Union[ImagesReference, List[ImagesReference]],
        class_names: List[str],
        model_version: Optional[str] = None,
        confidence: Optional[float] = None,
    ) -> List[dict]:
        """Run inference using YOLO-World model.

        Args:
            inference_input: Input image(s) to run inference on. Can be a single image
                reference or a list of image references.
            class_names: List of class names to detect in the image(s).
            model_version: Optional version of YOLO-World model to use. If not specified,
                uses the default version.
            confidence: Optional confidence threshold for detections. If not specified,
                uses the model's default threshold.

        Returns:
            List of dictionaries containing detection results for each input image.
            Each dictionary contains bounding boxes, class labels, and confidence scores
            for detected objects.

        Raises:
            HTTPCallErrorError: If there is an error in the HTTP call.
            HTTPClientError: If there is an error with the server connection.
        """
        encoded_inference_inputs = load_static_inference_input(
            inference_input=inference_input,
        )
        payload = self.__initialise_payload()
        payload["text"] = class_names
        if model_version is not None:
            payload["yolo_world_version_id"] = model_version
        if confidence is not None:
            payload["confidence"] = confidence
        url = self.__wrap_url_with_api_key(f"{self.__api_url}/yolo_world/infer")
        requests_data = prepare_requests_data(
            url=url,
            encoded_inference_inputs=encoded_inference_inputs,
            headers=DEFAULT_HEADERS,
            parameters=None,
            payload=payload,
            max_batch_size=1,
            image_placement=ImagePlacement.JSON,
        )
        responses = execute_requests_packages(
            requests_data=requests_data,
            request_method=RequestMethod.POST,
            max_concurrent_requests=self.__inference_configuration.max_concurrent_requests,
        )
        return [r.json() for r in responses]

    @wrap_errors_async
    async def infer_from_yolo_world_async(
        self,
        inference_input: Union[ImagesReference, List[ImagesReference]],
        class_names: List[str],
        model_version: Optional[str] = None,
        confidence: Optional[float] = None,
    ) -> List[dict]:
        """Run inference using YOLO-World model asynchronously.

        Args:
            inference_input: Input image(s) to run inference on. Can be a single image
                reference or a list of image references.
            class_names: List of class names to detect in the image(s).
            model_version: Optional version of YOLO-World model to use. If not specified,
                uses the default version.
            confidence: Optional confidence threshold for detections. If not specified,
                uses the model's default threshold.

        Returns:
            List of dictionaries containing detection results for each input image.
            Each dictionary contains bounding boxes, class labels, and confidence scores
            for detected objects.

        Raises:
            HTTPCallErrorError: If there is an error in the HTTP call.
            HTTPClientError: If there is an error with the server connection.
        """
        encoded_inference_inputs = await load_static_inference_input_async(
            inference_input=inference_input,
        )
        payload = self.__initialise_payload()
        payload["text"] = class_names
        if model_version is not None:
            payload["yolo_world_version_id"] = model_version
        if confidence is not None:
            payload["confidence"] = confidence
        url = self.__wrap_url_with_api_key(f"{self.__api_url}/yolo_world/infer")
        requests_data = prepare_requests_data(
            url=url,
            encoded_inference_inputs=encoded_inference_inputs,
            headers=DEFAULT_HEADERS,
            parameters=None,
            payload=payload,
            max_batch_size=1,
            image_placement=ImagePlacement.JSON,
        )
        return await execute_requests_packages_async(
            requests_data=requests_data,
            request_method=RequestMethod.POST,
            max_concurrent_requests=self.__inference_configuration.max_concurrent_requests,
        )

    @experimental(
        info="Video processing in inference server is under development. Breaking changes are possible."
    )
    @wrap_errors
    def start_inference_pipeline_with_workflow(
        self,
        video_reference: Union[str, int, List[Union[str, int]]],
        workflow_specification: Optional[dict] = None,
        workspace_name: Optional[str] = None,
        workflow_id: Optional[str] = None,
        image_input_name: str = "image",
        workflows_parameters: Optional[Dict[str, Any]] = None,
        workflows_thread_pool_workers: int = 4,
        cancel_thread_pool_tasks_on_exit: bool = True,
        video_metadata_input_name: str = "video_metadata",
        max_fps: Optional[Union[float, int]] = None,
        source_buffer_filling_strategy: Optional[BufferFillingStrategy] = "DROP_OLDEST",
        source_buffer_consumption_strategy: Optional[
            BufferConsumptionStrategy
        ] = "EAGER",
        video_source_properties: Optional[Dict[str, float]] = None,
        batch_collection_timeout: Optional[float] = None,
        results_buffer_size: int = 64,
    ) -> dict:
        """Starts an inference pipeline using a workflow specification.

        Args:
            video_reference: Path to video file, camera index, or list of video sources.
                Can be a string path, integer camera index, or list of either.
            workflow_specification: Optional workflow specification dictionary. Mutually
                exclusive with workspace_name/workflow_id.
            workspace_name: Optional name of workspace containing workflow. Must be used
                with workflow_id.
            workflow_id: Optional ID of workflow to use. Must be used with workspace_name.
            image_input_name: Name of the image input node in workflow. Defaults to "image".
            workflows_parameters: Optional parameters to pass to workflow.
            workflows_thread_pool_workers: Number of worker threads for workflow execution.
                Defaults to 4.
            cancel_thread_pool_tasks_on_exit: Whether to cancel pending tasks when exiting.
                Defaults to True.
            video_metadata_input_name: Name of video metadata input in workflow.
                Defaults to "video_metadata".
            max_fps: Optional maximum FPS to process video at.
            source_buffer_filling_strategy: Strategy for filling source buffer when full.
                One of: "WAIT", "DROP_OLDEST", "ADAPTIVE_DROP_OLDEST", "DROP_LATEST",
                "ADAPTIVE_DROP_LATEST". Defaults to "DROP_OLDEST".
            source_buffer_consumption_strategy: Strategy for consuming from source buffer.
                One of: "LAZY", "EAGER". Defaults to "EAGER".
            video_source_properties: Optional dictionary of video source properties.
            batch_collection_timeout: Optional timeout for batch collection in seconds.
            results_buffer_size: Size of results buffer. Defaults to 64.

        Returns:
            dict: Response containing pipeline initialization details.

        Raises:
            InvalidParameterError: If workflow specification parameters are invalid.
            HTTPCallErrorError: If there is an error in the HTTP call.
            HTTPClientError: If there is an error with the server connection.
        """
        named_workflow_specified = (workspace_name is not None) and (
            workflow_id is not None
        )
        if not (named_workflow_specified != (workflow_specification is not None)):
            raise InvalidParameterError(
                "Parameters (`workspace_name`, `workflow_id`) can be used mutually exclusive with "
                "`workflow_specification`, but at least one must be set."
            )
        payload = {
            "api_key": self.__api_key,
            "video_configuration": {
                "type": "VideoConfiguration",
                "video_reference": video_reference,
                "max_fps": max_fps,
                "source_buffer_filling_strategy": source_buffer_filling_strategy,
                "source_buffer_consumption_strategy": source_buffer_consumption_strategy,
                "video_source_properties": video_source_properties,
                "batch_collection_timeout": batch_collection_timeout,
            },
            "processing_configuration": {
                "type": "WorkflowConfiguration",
                "workflow_specification": workflow_specification,
                "workspace_name": workspace_name,
                "workflow_id": workflow_id,
                "image_input_name": image_input_name,
                "workflows_parameters": workflows_parameters,
                "workflows_thread_pool_workers": workflows_thread_pool_workers,
                "cancel_thread_pool_tasks_on_exit": cancel_thread_pool_tasks_on_exit,
                "video_metadata_input_name": video_metadata_input_name,
            },
            "sink_configuration": {
                "type": "MemorySinkConfiguration",
                "results_buffer_size": results_buffer_size,
            },
        }
        response = requests.post(
            f"{self.__api_url}/inference_pipelines/initialise",
            json=payload,
        )
        response.raise_for_status()
        return response.json()

    @experimental(
        info="Video processing in inference server is under development. Breaking changes are possible."
    )
    @wrap_errors
    def list_inference_pipelines(self) -> List[dict]:
        """Lists all active inference pipelines on the server.

        This method retrieves information about all currently running inference pipelines
        on the server, including their IDs and status.

        Returns:
            List[dict]: A list of dictionaries containing information about each active
                inference pipeline.

        Raises:
            HTTPCallErrorError: If there is an error in the HTTP call.
            HTTPClientError: If there is an error with the server connection.
        """
        payload = {"api_key": self.__api_key}
        response = requests.get(
            f"{self.__api_url}/inference_pipelines/list",
            json=payload,
        )
        api_key_safe_raise_for_status(response=response)
        return response.json()

    @experimental(
        info="Video processing in inference server is under development. Breaking changes are possible."
    )
    @wrap_errors
    def get_inference_pipeline_status(self, pipeline_id: str) -> dict:
        """Gets the current status of a specific inference pipeline.

        Args:
            pipeline_id: The unique identifier of the inference pipeline to check.

        Returns:
            dict: A dictionary containing the current status and details of the pipeline.

        Raises:
            HTTPCallErrorError: If there is an error in the HTTP call.
            HTTPClientError: If there is an error with the server connection.
            ValueError: If pipeline_id is empty or None.
        """
        self._ensure_pipeline_id_not_empty(pipeline_id=pipeline_id)
        payload = {"api_key": self.__api_key}
        response = requests.get(
            f"{self.__api_url}/inference_pipelines/{pipeline_id}/status",
            json=payload,
        )
        api_key_safe_raise_for_status(response=response)
        return response.json()

    @experimental(
        info="Video processing in inference server is under development. Breaking changes are possible."
    )
    @wrap_errors
    def pause_inference_pipeline(self, pipeline_id: str) -> dict:
        """Pauses a running inference pipeline.

        Sends a request to pause the specified inference pipeline. The pipeline must be
        currently running for this operation to succeed.

        Args:
            pipeline_id: The unique identifier of the inference pipeline to pause.

        Returns:
            dict: A dictionary containing the response from the server about the pause operation.

        Raises:
            HTTPCallErrorError: If there is an error in the HTTP call.
            HTTPClientError: If there is an error with the server connection.
            ValueError: If pipeline_id is empty or None.
        """
        self._ensure_pipeline_id_not_empty(pipeline_id=pipeline_id)
        payload = {"api_key": self.__api_key}
        response = requests.post(
            f"{self.__api_url}/inference_pipelines/{pipeline_id}/pause",
            json=payload,
        )
        api_key_safe_raise_for_status(response=response)
        return response.json()

    @experimental(
        info="Video processing in inference server is under development. Breaking changes are possible."
    )
    @wrap_errors
    def resume_inference_pipeline(self, pipeline_id: str) -> dict:
        """Resumes a paused inference pipeline.

        Sends a request to resume the specified inference pipeline. The pipeline must be
        currently paused for this operation to succeed.

        Args:
            pipeline_id: The unique identifier of the inference pipeline to resume.

        Returns:
            dict: A dictionary containing the response from the server about the resume operation.

        Raises:
            HTTPCallErrorError: If there is an error in the HTTP call.
            HTTPClientError: If there is an error with the server connection.
            ValueError: If pipeline_id is empty or None.
        """
        self._ensure_pipeline_id_not_empty(pipeline_id=pipeline_id)
        payload = {"api_key": self.__api_key}
        response = requests.post(
            f"{self.__api_url}/inference_pipelines/{pipeline_id}/resume",
            json=payload,
        )
        api_key_safe_raise_for_status(response=response)
        return response.json()

    @experimental(
        info="Video processing in inference server is under development. Breaking changes are possible."
    )
    @wrap_errors
    def terminate_inference_pipeline(self, pipeline_id: str) -> dict:
        """Terminates a running inference pipeline.

        Sends a request to terminate the specified inference pipeline. This will stop all
        processing and free up associated resources.

        Args:
            pipeline_id: The unique identifier of the inference pipeline to terminate.

        Returns:
            dict: A dictionary containing the response from the server about the termination operation.

        Raises:
            HTTPCallErrorError: If there is an error in the HTTP call.
            HTTPClientError: If there is an error with the server connection.
            ValueError: If pipeline_id is empty or None.
        """
        self._ensure_pipeline_id_not_empty(pipeline_id=pipeline_id)
        payload = {"api_key": self.__api_key}
        response = requests.post(
            f"{self.__api_url}/inference_pipelines/{pipeline_id}/terminate",
            json=payload,
        )
        api_key_safe_raise_for_status(response=response)
        return response.json()

    @experimental(
        info="Video processing in inference server is under development. Breaking changes are possible."
    )
    @wrap_errors
    def consume_inference_pipeline_result(
        self,
        pipeline_id: str,
        excluded_fields: Optional[List[str]] = None,
    ) -> dict:
        """Consumes and returns the next available result from an inference pipeline.

        Args:
            pipeline_id: The unique identifier of the inference pipeline to consume results from.
            excluded_fields: Optional list of field names to exclude from the result. If None,
                no fields will be excluded.

        Returns:
            dict: A dictionary containing the next available result from the pipeline.

        Raises:
            HTTPCallErrorError: If there is an error in the HTTP call.
            HTTPClientError: If there is an error with the server connection.
            InvalidParameterError: If pipeline_id is empty or None.
        """
        self._ensure_pipeline_id_not_empty(pipeline_id=pipeline_id)
        if excluded_fields is None:
            excluded_fields = []
        payload = {"api_key": self.__api_key, "excluded_fields": excluded_fields}
        response = requests.get(
            f"{self.__api_url}/inference_pipelines/{pipeline_id}/consume",
            json=payload,
        )
        api_key_safe_raise_for_status(response=response)
        return response.json()

    def _ensure_pipeline_id_not_empty(self, pipeline_id: str) -> None:
        if not pipeline_id:
            raise InvalidParameterError("Empty `pipeline_id` parameter detected")

    def _post_images(
        self,
        inference_input: Union[ImagesReference, List[ImagesReference]],
        endpoint: str,
        model_id: Optional[str] = None,
        extra_payload: Optional[Dict[str, Any]] = None,
    ) -> Union[dict, List[dict]]:
        encoded_inference_inputs = load_static_inference_input(
            inference_input=inference_input,
        )
        payload = self.__initialise_payload()
        if model_id is not None:
            payload["model_id"] = model_id
        url = self.__wrap_url_with_api_key(f"{self.__api_url}{endpoint}")
        if extra_payload is not None:
            payload.update(extra_payload)
        requests_data = prepare_requests_data(
            url=url,
            encoded_inference_inputs=encoded_inference_inputs,
            headers=DEFAULT_HEADERS,
            parameters=None,
            payload=payload,
            max_batch_size=self.__inference_configuration.max_batch_size,
            image_placement=ImagePlacement.JSON,
        )
        responses = execute_requests_packages(
            requests_data=requests_data,
            request_method=RequestMethod.POST,
            max_concurrent_requests=self.__inference_configuration.max_concurrent_requests,
        )
        results = [r.json() for r in responses]
        return unwrap_single_element_list(sequence=results)

    async def _post_images_async(
        self,
        inference_input: Union[ImagesReference, List[ImagesReference]],
        endpoint: str,
        model_id: Optional[str] = None,
        extra_payload: Optional[Dict[str, Any]] = None,
    ) -> Union[dict, List[dict]]:
        encoded_inference_inputs = await load_static_inference_input_async(
            inference_input=inference_input,
        )
        payload = self.__initialise_payload()
        if model_id is not None:
            payload["model_id"] = model_id
        url = self.__wrap_url_with_api_key(f"{self.__api_url}{endpoint}")
        if extra_payload is not None:
            payload.update(extra_payload)
        requests_data = prepare_requests_data(
            url=url,
            encoded_inference_inputs=encoded_inference_inputs,
            headers=DEFAULT_HEADERS,
            parameters=None,
            payload=payload,
            max_batch_size=self.__inference_configuration.max_batch_size,
            image_placement=ImagePlacement.JSON,
        )
        responses = await execute_requests_packages_async(
            requests_data=requests_data,
            request_method=RequestMethod.POST,
            max_concurrent_requests=self.__inference_configuration.max_concurrent_requests,
        )
        return unwrap_single_element_list(sequence=responses)

    def __initialise_payload(self) -> dict:
        if self.__client_mode is not HTTPClientMode.V0:
            return {"api_key": self.__api_key}
        return {}

    def __wrap_url_with_api_key(self, url: str) -> str:
        if self.__client_mode is not HTTPClientMode.V0:
            return url
        return f"{url}?api_key={self.__api_key}"

    def __ensure_v1_client_mode(self) -> None:
        if self.__client_mode is not HTTPClientMode.V1:
            raise WrongClientModeError("Use client mode `v1` to run this operation.")


def _determine_client_downsizing_parameters(
    client_downsizing_disabled: bool,
    model_description: Optional[ModelDescription],
    default_max_input_size: int,
) -> Tuple[Optional[int], Optional[int]]:
    if client_downsizing_disabled:
        return None, None
    if (
        model_description is None
        or model_description.input_height is None
        or model_description.input_width is None
    ):
        return default_max_input_size, default_max_input_size
    return model_description.input_height, model_description.input_width


def _determine_client_mode(api_url: str) -> HTTPClientMode:
    if any(api_url.startswith(roboflow_url) for roboflow_url in ALL_ROBOFLOW_API_URLS):
        return HTTPClientMode.V0
    return HTTPClientMode.V1


def _ensure_model_is_selected(model_id: Optional[str]) -> None:
    if model_id is None:
        raise ModelNotSelectedError("No model was selected to be used.")


def _ensure_api_key_provided(api_key: Optional[str]) -> None:
    if api_key is None:
        raise APIKeyNotProvided("API key must be provided in this case")
