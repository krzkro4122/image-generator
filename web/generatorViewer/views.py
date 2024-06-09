from django.http import Http404, HttpRequest, HttpResponse
from django.template import loader

from generatorViewer.src.views_utils import (
    load_generated_images,
    create_images,
    Keys,
)


def index(request: HttpRequest):
    batch_size = 8  # TODO - Get from UI in the request
    batch_index = 0  # TODO - Get from a DB or local storage

    request_model_name = request.POST.get(Keys.MODEL_NAME)
    model_name = "celeba" if not request_model_name else request_model_name

    image_tensors = load_generated_images(batch_size=batch_size, model_name=model_name)

    return HttpResponse(
        loader.get_template("images.html").render(
            request=request,
            context={
                Keys.IMAGES: create_images(
                    batch_index=batch_index,
                    batch_size=batch_size,
                    image_tensors=image_tensors,
                    model_name=model_name,
                ),
            },
        )
    )


def next_batch(request: HttpRequest):
    if request.method != "POST":
        raise Http404("Method not allowed")

    request_batch_size = int(request.POST[Keys.BATCH_SIZE])

    request_model_name = request.POST.get(Keys.MODEL_NAME)
    model_name = "celeba" if not request_model_name else request_model_name

    image_tensors = load_generated_images(
        batch_size=request_batch_size,
        model_name=model_name,
    )

    return HttpResponse(
        loader.get_template("images_form.html").render(
            request=request,
            context={
                Keys.IMAGES: create_images(
                    batch_size=request_batch_size,
                    image_tensors=image_tensors,
                    model_name=model_name,
                ),
            },
        )
    )
